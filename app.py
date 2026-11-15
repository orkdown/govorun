import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import queue
import re
import socket
import sys
import threading
import time
from typing import List, Optional, Union

from num2words import num2words

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import scipy.signal as signal
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sounddevice as sd
import torch
import uvicorn
import discord
from discord.ext import commands

if not discord.opus.is_loaded():
    try:
        discord.opus._load_default()
    except Exception:
        pass

# --- КОНФИГУРАЦИЯ ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "tg_token": "",
    "tg_user_id": 0,
    "discord_token": "",
    "discord_voice_channel_id": None,
    "discord_auto_connect": True,
    "output_cable_enabled": True,
    "output_discord_enabled": True,
    "speaker": "aidar",
    "effect": "none",
    "cable_device_id": None,
    "monitor_device_id": None,
    "monitor_enabled": True,
    "volume": 1.0,
    "sfx_volume": 0.5,
    "quick_phrases": [
        "👍 Да, согласен",
        "👎 Неа, не выйдет",
        "🤫 Парни, я без микрофона, пишу с пульта",
        "☕ Отойду на 5 минут, скоро буду",
        "😂 Ахаха, хорош!",
        "👀 Секунду, сейчас посмотрю",
        "👋 Всем привет!",
        "🎮 Го в катку!",
        "🏆 Красиво сыграно, GG!"
    ]
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TTSBridge")


def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg.update(saved)
        except Exception as e:
            logger.warning(f"Ошибка чтения {CONFIG_FILE}, используются дефолтные: {e}")
    else:
        save_config(cfg)
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Не удалось сохранить {CONFIG_FILE}: {e}")


config = load_config()


# --- АВТОПОИСК АУДИОУСТРОЙСТВ ---
def find_cable_device():
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if "CABLE Input" in dev["name"] and dev["max_output_channels"] > 0:
                return idx
        default_out = sd.default.device[1]
        return default_out if default_out is not None and default_out >= 0 else 0
    except Exception as e:
        logger.info(f"Локальные звуковые устройства не обнаружены (headless/серверный режим): {e}")
        return None


def find_default_monitor_device():
    try:
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name_lower = dev["name"].lower()
            if ("headphone" in name_lower or "наушник" in name_lower or "speaker" in name_lower or "динамик" in name_lower) and dev["max_output_channels"] > 0:
                if "cable" not in name_lower:
                    return idx
        default_out = sd.default.device[1]
        return default_out if default_out is not None and default_out >= 0 else 0
    except Exception as e:
        logger.info(f"Устройства мониторинга не обнаружены (headless/серверный режим): {e}")
        return None


if config["cable_device_id"] is None:
    config["cable_device_id"] = find_cable_device()
    save_config(config)

if config["monitor_device_id"] is None:
    config["monitor_device_id"] = find_default_monitor_device()
    save_config(config)


# --- ЗАГРУЗКА И ИНИЦИАЛИЗАЦИЯ SILERO TTS ---
print("\n" + "=" * 60)
print(" [1/3] Загрузка нейросети Silero TTS (snakers4/silero-models)...")
print("=" * 60)

torch_threads = min(8, max(2, (os.cpu_count() or 4) - 1))
torch.set_num_threads(torch_threads)
device = torch.device("cpu")

try:
    hub_dir = torch.hub.get_dir()
    cached_repo = os.path.join(hub_dir, 'snakers4_silero-models_master')
    if os.path.exists(cached_repo):
        logger.info(f"Загрузка Silero TTS из локального кэша: {cached_repo}")
        model, _ = torch.hub.load(
            repo_or_dir=cached_repo,
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            source="local"
        )
    else:
        logger.info("Локальный кэш не найден, загрузка Silero TTS с GitHub...")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True
        )
    model.to(device)
    print(" [2/3] Модель успешно загружена!")
    print(" [3/3] Предпрогрев (Warm-up) модели для мгновенного отклика (<300ms)...")
    _ = model.apply_tts(text="готов", speaker="aidar", sample_rate=48000, put_accent=True, put_yo=True)
    print(" [OK] Система синтеза речи полностью прогрета и готова!")
except Exception as e:
    logger.exception(f"Критическая ошибка загрузки модели Silero: {e}")
    sys.exit(1)


# --- ПОТОКОБЕЗОПАСНАЯ ОЧЕРЕДЬ TTS И МИКШИРОВАНИЕ ---
tts_queue = queue.Queue()
status_state = {
    "status": "idle",
    "current_text": "",
    "speaker": config["speaker"],
    "queue_len": 0
}


# --- ЗВУКОВЫЕ ЭФФЕКТЫ (DSP / БЫСТРЫЕ ПРЕСЕТЫ) ---
def pitch_shift(x: np.ndarray, semitones: float, sr: int = 48000) -> np.ndarray:
    factor = 2.0 ** (semitones / 12.0)
    win_size = int(0.03 * sr)
    hop_out = win_size // 2
    # Для повышения питча (semitones > 0) hop_in должен быть меньше hop_out:
    hop_in = int(round(hop_out / factor))
    if hop_in <= 0:
        return x
    num_hops = (len(x) - win_size) // hop_in
    if num_hops <= 0:
        return x
    y = np.zeros(num_hops * hop_out + win_size, dtype=np.float32)
    win = np.hanning(win_size).astype(np.float32)
    for i in range(num_hops):
        in_pos = i * hop_in
        out_pos = i * hop_out
        y[out_pos:out_pos + win_size] += x[in_pos:in_pos + win_size] * win
    # Качественный ресемплинг Фурье без замыливания частот:
    return signal.resample(y, len(x)).astype(np.float32)


def apply_audio_effect(audio: np.ndarray, effect: str, sample_rate: int = 48000) -> np.ndarray:
    if not effect or effect == "none":
        return audio
    try:
        if effect == "radio":
            sos = signal.butter(4, [350, 3200], btype="bandpass", fs=sample_rate, output="sos")
            filtered = signal.sosfilt(sos, audio)
            driven = np.tanh(filtered * 2.2) * 0.85
            return driven.astype(np.float32)
        elif effect == "chipmunk":
            # Бурундук / Гном: высокий питч (+5 полутонов)
            return pitch_shift(audio, 5.0, sample_rate)
        elif effect == "demon":
            # Демон / Монстр: низкий питч (-5 полутонов) + басовый низ
            shifted = pitch_shift(audio, -5.0, sample_rate)
            sos_bass = signal.butter(2, 300, btype="lowpass", fs=sample_rate, output="sos")
            bass = signal.sosfilt(sos_bass, audio) * 0.4
            combined = shifted * 0.85 + bass
            return np.clip(combined, -1.0, 1.0).astype(np.float32)
        elif effect == "robot":
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 65.0 * np.arange(len(audio)) / sample_rate)
            return (audio * mod * 1.1).astype(np.float32)
        elif effect == "echo":
            # Пространственное затухающее эхо / арена (многократное отражение)
            delays_ms = [75, 150, 240, 340, 450]
            gains = [0.35, 0.22, 0.14, 0.08, 0.04]
            out = np.copy(audio)
            for d_ms, g in zip(delays_ms, gains):
                d_samp = int(d_ms * sample_rate / 1000)
                if len(audio) > d_samp:
                    out[d_samp:] += audio[:-d_samp] * g
            return np.clip(out, -1.0, 1.0).astype(np.float32)
    except Exception as e:
        logger.error(f"Ошибка применения эффекта {effect}: {e}")
    return audio


# --- ДВИЖОК PIPER TTS (VITS) ---
piper_voices_cache = {}


def get_piper_voice(voice_name: str):
    key = "dmitri" if "dmitri" in voice_name else "irina"
    if key in piper_voices_cache:
        return piper_voices_cache[key]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "piper_models", key)
    fn = "ru_RU-dmitri-medium.onnx" if key == "dmitri" else "ru_RU-irina-medium.onnx"
    model_path = os.path.join(model_dir, fn)
    config_path = model_path + ".json"
    if os.path.exists(model_path) and os.path.exists(config_path):
        try:
            from piper.voice import PiperVoice
            logger.info(f"Загрузка движка Piper ({key})...")
            v = PiperVoice.load(model_path, config_path)
            piper_voices_cache[key] = v
            return v
        except Exception as e:
            logger.error(f"Ошибка загрузки Piper {key}: {e}")
    return None


def synthesize_piper(text: str, voice_name: str) -> np.ndarray:
    voice = get_piper_voice(voice_name)
    if voice is None:
        return model.apply_tts(text=text, speaker="aidar", sample_rate=48000, put_accent=True, put_yo=True).numpy()
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio_float = np.concatenate([c.audio_float_array for c in chunks])
    sr = chunks[0].sample_rate
    if sr != 48000 and len(audio_float) > 0:
        target_len = int(len(audio_float) * 48000 / sr)
        audio_float = signal.resample(audio_float, target_len).astype(np.float32)
    return audio_float


# --- ДВИЖОК EDGE TTS (MICROSOFT NEURAL) ---
def synthesize_edge_tts(text: str, voice_name: str = "ru-RU-DmitryNeural") -> np.ndarray:
    import io
    import time
    import edge_tts
    import soundfile as sf

    async def _generate():
        comm = edge_tts.Communicate(text, voice_name, rate="+0%", pitch="+0Hz")
        buf = io.BytesIO()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return buf.getvalue()

    for attempt in range(1, 4):
        try:
            mp3_bytes = asyncio.run(_generate())
            if mp3_bytes and len(mp3_bytes) > 0:
                with io.BytesIO(mp3_bytes) as f:
                    data, sr = sf.read(f, dtype="float32")
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if sr != 48000 and len(data) > 0:
                    target_len = int(len(data) * 48000 / sr)
                    data = signal.resample(data, target_len).astype(np.float32)
                return data
            logger.warning(f"Edge-TTS ({voice_name}) попытка {attempt}/3 вернула 0 байт, повторяем...")
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Edge-TTS ({voice_name}) попытка {attempt}/3 не удалась: {e}")
            time.sleep(0.3)

    logger.error(f"Сбой соединения с серверами Microsoft Edge TTS ({voice_name}). Автопереход на оффлайн-голос Piper (dmitri)...")
    return synthesize_piper(text, "dmitri")


def decode_audio_bytes(file_bytes: bytes) -> np.ndarray:
    import io
    import soundfile as sf
    with io.BytesIO(file_bytes) as f:
        data, sr = sf.read(f, dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != 48000 and len(data) > 0:
        target_len = int(len(data) * 48000 / sr)
        data = signal.resample(data, target_len).astype(np.float32)
    return data


# --- АУДИОПОТОК ДЛЯ DISCORD ГОЛОСОВОГО КАНАЛА ---
def prepare_discord_pcm(audio_data: np.ndarray, samplerate: int = 48000) -> bytes:
    import io
    data = np.asarray(audio_data, dtype=np.float32)
    if data.ndim == 2:
        if data.shape[1] == 1:
            data = data.squeeze(axis=1)
        elif data.shape[0] == 1:
            data = data.squeeze(axis=0)
        elif data.shape[1] == 2:
            pass
        else:
            data = data.mean(axis=1)

    if data.ndim == 1:
        stereo = np.column_stack((data, data))
    else:
        stereo = data

    if samplerate != 48000 and len(stereo) > 0:
        target_len = int(len(stereo) * 48000 / samplerate)
        stereo = signal.resample(stereo, target_len, axis=0)

    stereo = np.clip(stereo, -1.0, 1.0)
    int16_data = (stereo * 32767.0).astype(np.int16)
    if not int16_data.flags.c_contiguous:
        int16_data = np.ascontiguousarray(int16_data)

    raw_bytes = int16_data.tobytes()
    # 4 фрейма (80 мс) тишины до и после для открытия джиттер-буфера Discord
    silence_frame = b'\x00' * 3840
    return (silence_frame * 4) + raw_bytes + (silence_frame * 4)


class MemoryPCMAudio(discord.AudioSource):
    def __init__(self, audio_data: np.ndarray, samplerate: int = 48000):
        import io
        pcm_bytes = prepare_discord_pcm(audio_data, samplerate)
        self.stream = io.BytesIO(pcm_bytes)
        self.frame_size = 3840  # 20ms стерео 16-бит при 48 кГц

    def read(self) -> bytes:
        data = self.stream.read(self.frame_size)
        return data if len(data) == self.frame_size else b""

    def is_opus(self) -> bool:
        return False


active_discord_voice_client: Optional[discord.VoiceClient] = None


def get_active_voice_client() -> Optional[discord.VoiceClient]:
    global active_discord_voice_client
    if active_discord_voice_client and active_discord_voice_client.is_connected():
        return active_discord_voice_client
    if discord_bot_instance and discord_bot_instance.voice_clients:
        for vc in discord_bot_instance.voice_clients:
            if vc.is_connected():
                active_discord_voice_client = vc
                return vc
    return None


def is_discord_voice_connected() -> bool:
    vc = get_active_voice_client()
    return vc is not None and vc.is_connected()


def play_to_discord_bot(audio_data: np.ndarray):
    vc = get_active_voice_client()
    if not vc or not vc.is_connected():
        return
    try:
        wait_s = 0.0
        while vc.is_playing() and wait_s < 30.0:
            time.sleep(0.05)
            wait_s += 0.05

        import tempfile
        import soundfile as sf

        data = np.asarray(audio_data, dtype=np.float32)
        if data.ndim == 1:
            stereo = np.column_stack((data, data))
        elif data.ndim == 2 and data.shape[1] == 1:
            stereo = np.column_stack((data[:, 0], data[:, 0]))
        else:
            stereo = data

        stereo = np.clip(stereo, -1.0, 1.0)
        silence_samples = int(48000 * 0.1)
        silence = np.zeros((silence_samples, 2), dtype=np.float32)
        padded_audio = np.concatenate([silence, stereo, silence], axis=0)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp_path = f.name

        sf.write(tmp_path, padded_audio, 48000, subtype='PCM_16')

        try:
            src = discord.FFmpegPCMAudio(tmp_path, options="-vn -loglevel error")
        except Exception:
            src = discord.FFmpegOpusAudio(tmp_path)

        def _after_play(err):
            try:
                src.cleanup()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            if err:
                logger.error(f"[Discord Voice] Ошибка в плеере: {err}")
            else:
                logger.info("[Discord Voice] Аудио успешно передано в голосовой канал!")

        logger.info(f"[Discord Voice] Воспроизведение в '{vc.channel.name}' через FFmpeg (is_opus: {src.is_opus()})...")
        vc.play(src, after=_after_play)
        while vc.is_playing():
            time.sleep(0.05)
    except Exception as e:
        logger.error(f"[Discord Voice] Сбой при воспроизведении: {e}")


def play_to_device(audio_data: np.ndarray, dev_id: int, samplerate: int):
    if dev_id is None:
        return
    try:
        channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
        with sd.OutputStream(device=dev_id, samplerate=samplerate, channels=channels, dtype=np.float32) as stream:
            stream.write(audio_data)
    except Exception as e:
        logger.error(f"Ошибка воспроизведения на устройстве {dev_id}: {e}")


def worker_loop():
    logger.info("Фоновый воркер очереди озвучки запущен.")
    while True:
        task = tts_queue.get()
        if task is None:
            break
        text, spk, volume, eff, precomputed_audio = task
        try:
            if precomputed_audio is not None:
                status_state["status"] = "speaking"
                status_state["current_text"] = text or "[Аудиофайл]"
                audio_np = precomputed_audio
            else:
                status_state["status"] = "generating"
                status_state["current_text"] = text
                status_state["speaker"] = spk
                status_state["queue_len"] = tts_queue.qsize()

                # Выбор синтезатора: Edge-TTS, Piper или Silero
                if spk in ["edge_dmitry", "dmitry_neural", "ru-RU-DmitryNeural", "edge_dmitri", "edge"]:
                    audio_np = synthesize_edge_tts(text, "ru-RU-DmitryNeural")
                elif spk in ["dmitri", "irina", "piper_dmitri", "piper_irina"]:
                    audio_np = synthesize_piper(text, spk)
                else:
                    actual_spk = spk if spk in model.speakers else "aidar"
                    audio_tensor = model.apply_tts(
                        text=text,
                        speaker=actual_spk,
                        sample_rate=48000,
                        put_accent=True,
                        put_yo=True
                    )
                    audio_np = audio_tensor.numpy()

                if eff and eff != "none":
                    audio_np = apply_audio_effect(audio_np, eff, 48000)

            if volume != 1.0:
                audio_np = np.clip(audio_np * volume, -1.0, 1.0)

            status_state["status"] = "speaking"

            threads = []
            cable_id = config.get("cable_device_id")
            if cable_id is not None and config.get("output_cable_enabled", True):
                t1 = threading.Thread(target=play_to_device, args=(audio_np, cable_id, 48000), daemon=True)
                threads.append(t1)
                t1.start()

            if config.get("monitor_enabled") and config.get("monitor_device_id") is not None:
                monitor_id = config.get("monitor_device_id")
                if monitor_id != cable_id:
                    t2 = threading.Thread(target=play_to_device, args=(audio_np, monitor_id, 48000), daemon=True)
                    threads.append(t2)
                    t2.start()

            if config.get("output_discord_enabled", True) and is_discord_voice_connected():
                t3 = threading.Thread(target=play_to_discord_bot, args=(audio_np,), daemon=True)
                threads.append(t3)
                t3.start()

            for t in threads:
                t.join()

        except Exception as e:
            logger.error(f"Ошибка воспроизведения: {e}")
        finally:
            status_state["status"] = "idle"
            status_state["current_text"] = ""
            status_state["queue_len"] = tts_queue.qsize()
            tts_queue.task_done()


worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()


# --- НОРМАЛИЗАЦИЯ ТЕКСТА (ЧИСЛА ПРОПИСЬЮ И АНГЛИЙСКИЙ ТРАНСЛИТ) ---
COMMON_EN = {
    'discord': 'дискорд', 'gg': 'джи джи', 'wp': 'вэ пэ', 'ez': 'изи',
    'afk': 'афк', 'pls': 'плиз', 'plz': 'плиз', 'thx': 'спс', 'ty': 'спасибо',
    'hi': 'хай', 'hello': 'хеллоу', 'bye': 'бай', 'go': 'го', 'ok': 'ок',
    'okay': 'окей', 'yes': 'да', 'no': 'ноу', 'bro': 'бро', 'cs': 'кээс',
    'cs2': 'кээс два', 'dota': 'дота', 'steam': 'стим', 'hp': 'хэпэ',
    'xp': 'икспи', 'lvl': 'левел', 'ping': 'пинг', 'lag': 'лаг', 'lags': 'лаги',
    'admin': 'админ', 'fake': 'фейк', 'pro': 'про', 'noob': 'нуб', 'bot': 'бот',
    'team': 'тим', 'win': 'вин', 'lose': 'луз', 'gl': 'гуд лак', 'hf': 'хэв фан',
    'bb': 'бэ бэ', 'mute': 'мут', 'report': 'репорт', 'crash': 'краш',
    'play': 'плей', 'game': 'гейм', 'mic': 'микрофон', 'voice': 'войс',
    'stats': 'стата'
}

TRANSLIT_RULES = [
    ('shch', 'щ'), ('sh', 'ш'), ('ch', 'ч'), ('zh', 'ж'), ('th', 'т'),
    ('ph', 'ф'), ('kh', 'х'), ('ts', 'ц'), ('ya', 'я'), ('yu', 'ю'),
    ('yo', 'ё'), ('ee', 'и'), ('oo', 'у'), ('ck', 'к'), ('qu', 'кв')
]

CHAR_MAP = {
    'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
    'g': 'г', 'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л',
    'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
    's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
    'y': 'и', 'z': 'з'
}


def transliterate_word(word: str) -> str:
    w = word.lower()
    if w in COMMON_EN:
        return COMMON_EN[w]
    for eng, ru in TRANSLIT_RULES:
        w = w.replace(eng, ru)
    res = [CHAR_MAP.get(char, char) for char in w]
    return ''.join(res)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Знаки валют и проценты
    text = re.sub(r'(\d+)\s*%', r'\1 процентов', text)
    text = re.sub(r'(\d+)\s*\$', r'\1 долларов', text)
    text = re.sub(r'(\d+)\s*€', r'\1 евро', text)
    text = re.sub(r'(\d+)\s*₽', r'\1 рублей', text)
    # Знак + заменяем на слово 'плюс' ТОЛЬКО если после него НЕ идет русская гласная (для ударений в Silero)
    text = re.sub(r'\+(?![аеёиоуыэюяАЕЁИОУЫЭЮЯ])', ' плюс ', text)
    text = text.replace('=', ' равно ')

    # Преобразование чисел в слова на русском языке
    def replace_num(match):
        num_str = match.group(0)
        try:
            val = int(num_str)
            return num2words(val, lang='ru')
        except Exception:
            return num_str

    text = re.sub(r'\b\d+\b', replace_num, text)

    # Преобразование английских слов в русский фонетический транслит / сленг
    def replace_en(match):
        return transliterate_word(match.group(0))

    text = re.sub(r'\b[a-zA-Z]+\b', replace_en, text)
    return re.sub(r'\s+', ' ', text).strip()


def enqueue_tts(text: str, speaker: Optional[str] = None, effect: Optional[str] = None, raw_audio: Optional[np.ndarray] = None):
    if raw_audio is not None:
        sfx_vol = float(config.get("sfx_volume", 0.5))
        logger.info(f"Воспроизведение аудиофайла: {text} (громкость SFX: {int(sfx_vol * 100)}%)")
        tts_queue.put((text or "[Аудиофайл]", None, sfx_vol, None, raw_audio))
        status_state["queue_len"] = tts_queue.qsize()
        return

    raw = text.strip()
    if not raw:
        return
    cleaned = normalize_text(raw)
    if not cleaned:
        return
    spk = speaker or config.get("speaker", "aidar")
    eff = effect or config.get("effect", "none")
    vol = float(config.get("volume", 1.0))
    logger.info(f"Озвучивание: '{raw}' -> '{cleaned}' (голос: {spk}, эффект: {eff}, громкость: {int(vol * 100)}%)")
    tts_queue.put((cleaned, spk, vol, eff, None))
    status_state["queue_len"] = tts_queue.qsize()


# --- TELEGRAM BOT ИНТЕГРАЦИЯ (AIOGRAM 3.X) ---
tg_bot_task: Optional[asyncio.Task] = None
tg_bot_instance = None
tg_dispatcher_instance = None


async def start_telegram_bot_service():
    global tg_bot_instance, tg_dispatcher_instance
    token = config.get("tg_token", "").strip()
    if not token:
        logger.info("[Telegram] Токен не указан. Бот не запускается.")
        return

    try:
        from aiogram import Bot, Dispatcher, F, types
        from aiogram.filters import Command
        from aiogram.utils.keyboard import InlineKeyboardBuilder

        bot = Bot(token=token)
        dp = Dispatcher()
        tg_bot_instance = bot
        tg_dispatcher_instance = dp

        silero_speakers = [
            ("aidar", "⚡ Айдар (Мужской)"),
            ("eugene", "🎙️ Евгений (Баритон)"),
            ("xenia", "✨ Ксения (Четкий)"),
            ("baya", "🌸 Байя (Мягкий)"),
            ("kseniya", "🎭 Ксения 2 (Эмоц.)"),
            ("random", "🎲 Рандомный")
        ]

        piper_speakers = [
            ("dmitri", "⚡ Дмитрий (Piper VITS)"),
            ("irina", "🌸 Ирина (Piper VITS)")
        ]

        edge_speakers = [
            ("edge_dmitry", "🌐 Дмитрий Neural (Edge)")
        ]

        all_speakers_dict = dict(silero_speakers + piper_speakers + edge_speakers)

        available_effects = [
            ("none", "✨ Обычный"),
            ("radio", "📻 Рация"),
            ("chipmunk", "🐿️ Бурундук"),
            ("demon", "👹 Демон"),
            ("robot", "🤖 Робот"),
            ("echo", "📢 Эхо")
        ]

        def get_voice_keyboard():
            builder = InlineKeyboardBuilder()
            # Silero секция
            builder.row(types.InlineKeyboardButton(text="— 🗣️ Silero TTS (Эмоции/Ударения) —", callback_data="header_silero"))
            for code, name in silero_speakers:
                prefix = "✅ " if config.get("speaker") == code else ""
                builder.button(text=f"{prefix}{name}", callback_data=f"voice_{code}")
            builder.adjust(1, 2, 2, 2)

            # Piper секция
            builder.row(types.InlineKeyboardButton(text="— 🎤 Piper TTS (Студийные VITS) —", callback_data="header_piper"))
            for code, name in piper_speakers:
                prefix = "✅ " if config.get("speaker") == code else ""
                builder.button(text=f"{prefix}{name}", callback_data=f"voice_{code}")

            # Edge TTS секция
            builder.row(types.InlineKeyboardButton(text="— 🌐 Edge TTS (Microsoft Neural) —", callback_data="header_edge"))
            for code, name in edge_speakers:
                prefix = "✅ " if config.get("speaker") == code else ""
                builder.button(text=f"{prefix}{name}", callback_data=f"voice_{code}")

            mon_status = "🔊 Мониторинг: ВКЛ" if config.get("monitor_enabled") else "🔇 Мониторинг: ВЫКЛ"
            eff_name = dict(available_effects).get(config.get("effect", "none"), "✨ Обычный")
            builder.row(
                types.InlineKeyboardButton(text=mon_status, callback_data="toggle_monitor"),
                types.InlineKeyboardButton(text=f"FX: {eff_name}", callback_data="open_effects")
            )
            return builder.as_markup()

        def get_effect_keyboard():
            builder = InlineKeyboardBuilder()
            for code, name in available_effects:
                prefix = "✅ " if config.get("effect", "none") == code else ""
                builder.button(text=f"{prefix}{name}", callback_data=f"effect_{code}")
            builder.adjust(2)
            builder.row(types.InlineKeyboardButton(text="⬅️ Назад к голосам", callback_data="open_voices"))
            return builder.as_markup()

        @dp.message(Command("start"))
        async def cmd_start(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                await msg.answer(f"⛔ Доступ запрещен. Ваш Telegram ID: <code>{user_id}</code>\nУкажите его в панели управления на ПК.", parse_mode="HTML")
                return

            mon_text = "ВКЛ (слышно в наушниках)" if config.get("monitor_enabled") else "ВЫКЛ"
            eff_name = dict(available_effects).get(config.get("effect", "none"), "✨ Обычный")
            spk_name = all_speakers_dict.get(config.get("speaker"), config.get("speaker"))
            text = (
                f"🎙️ <b>TTS Discord Bridge (Silero + Piper + Edge)</b>\n\n"
                f"Бот подключен и готов озвучивать текст и играть звуки в Discord!\n\n"
                f"👤 Ваш Telegram ID: <code>{user_id}</code>\n"
                f"🗣️ Текущий голос: <b>{spk_name}</b>\n"
                f"🎭 Эффект: <b>{eff_name}</b>\n"
                f"🎧 Мониторинг: <b>{mon_text}</b>\n\n"
                f"💬 <b>Возможности:</b>\n"
                f"• Текст в чат ➡️ озвучка в микрофон Discord\n"
                f"• Аудиофайл или голосовое ➡️ воспроизведение в Discord\n"
                f"• /voice — выбор голоса (Silero / Piper / Edge TTS)\n"
                f"• /effect — звуковые эффекты (рация, демон, робот, эхо...)\n"
                f"• /monitor — переключение звука в наушниках\n"
                f"• /help — <b>шпаргалка по Silero, Piper, Edge TTS и тегам</b>\n"
                f"• /status — статус очереди и параметров\n"
                f"• /discord — управление ботом в голосовом канале Discord"
            )
            await msg.answer(text, parse_mode="HTML", reply_markup=get_voice_keyboard())

        @dp.message(Command("help"))
        async def cmd_help(msg: types.Message):
            text = (
                f"📖 <b>Шпаргалка: Silero vs Piper vs Edge TTS:</b>\n\n"
                f"🗣️ <b>1. Silero TTS (Айдар, Евгений, Ксения, Байя):</b>\n"
                f"  • Максимальная эмоциональность и отыгрыш\n"
                f"  • <b>Знак <code>+</code> перед гласной</b> — принудительное ударение:\n"
                f"    <code>Впереди старинный з+амок</code> (крепость)\n"
                f"    <code>На двери висит зам+ок</code> (дверной)\n"
                f"  • <b><code>...</code> (многоточие)</b> — пауза раздумья (~0.5 сек)\n"
                f"  • <b><code>—</code> (тире)</b> — перепад тона и смысловой акцент\n"
                f"  • <b><code>?!</code></b> — вопросительный и восклицательный подъем\n\n"
                f"🎤 <b>2. Piper TTS (Дмитрий, Ирина):</b>\n"
                f"  • Чистый оффлайн-голос диктора без артефактов\n"
                f"  • Автоматически читает любые числа, даты, дроби\n"
                f"  • Знак <code>+</code> ставить не нужно (работает на фонемах)\n\n"
                f"🌐 <b>3. Edge TTS (Дмитрий Neural от Microsoft):</b>\n"
                f"  • Естественный мужской нейро-голос ru-RU-DmitryNeural\n"
                f"  • Работает через облако Microsoft Edge (без задержек и без лимитов)\n"
                f"  • Идеально читает длинные предложения и сложные тексты\n\n"
                f"🎵 <b>4. Проигрывание аудиофайлов и музыки:</b>\n"
                f"  • Перешлите в этот бот любой аудиофайл (mp3, wav, ogg, flac) или голосовое сообщение — оно мгновенно прозвучит в Discord с громкостью SFX (50%)!\n\n"
                f"⚡ <b>5. Быстрые теги в сообщении:</b>\n"
                f"  • <code>[radio] Прием, как слышно?</code>\n"
                f"  • <code>[demon] Я уничтожу вас!</code>\n"
                f"  • <code>[edge_dmitry] Естественный диктор от Microsoft</code>\n"
                f"  • <code>[eugene:radio] Внимание отряду!</code>"
            )
            await msg.answer(text, parse_mode="HTML")

        @dp.message(Command("monitor"))
        async def cmd_monitor(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                return
            config["monitor_enabled"] = not config.get("monitor_enabled", False)
            save_config(config)
            st = "🔊 ВКЛЮЧЕН (слышно в наушниках)" if config["monitor_enabled"] else "🔇 ВЫКЛЮЧЕН"
            await msg.answer(f"🎧 <b>Мониторинг звука:</b> {st}", parse_mode="HTML", reply_markup=get_voice_keyboard())

        @dp.callback_query(F.data == "toggle_monitor")
        async def on_toggle_monitor(callback: types.CallbackQuery):
            user_id = callback.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                await callback.answer("Нет доступа", show_alert=True)
                return
            config["monitor_enabled"] = not config.get("monitor_enabled", False)
            save_config(config)
            await callback.message.edit_reply_markup(reply_markup=get_voice_keyboard())
            st = "ВКЛЮЧЕН" if config["monitor_enabled"] else "ВЫКЛЮЧЕН"
            await callback.answer(f"Мониторинг {st}")

        @dp.callback_query(F.data.startswith("header_"))
        async def on_header_clicked(callback: types.CallbackQuery):
            await callback.answer()

        @dp.message(Command("voice"))
        async def cmd_voice(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                return
            await msg.answer("🗣️ Выберите голос для озвучки в Discord:", reply_markup=get_voice_keyboard())

        @dp.callback_query(F.data == "open_voices")
        async def on_open_voices(callback: types.CallbackQuery):
            await callback.message.edit_text("🗣️ Выберите голос для озвучки:", reply_markup=get_voice_keyboard())

        @dp.callback_query(F.data.startswith("voice_"))
        async def on_voice_selected(callback: types.CallbackQuery):
            user_id = callback.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                await callback.answer("Нет доступа", show_alert=True)
                return

            selected = callback.data.split("_")[1]
            config["speaker"] = selected
            save_config(config)
            spk_name = all_speakers_dict.get(selected, selected)
            await callback.message.edit_text(
                f"✅ Голос изменен на: <b>{spk_name}</b>",
                parse_mode="HTML",
                reply_markup=get_voice_keyboard()
            )
            await callback.answer(f"Выбран: {selected}")

        @dp.message(Command("effect"), Command("fx"))
        async def cmd_effect(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                return
            await msg.answer("🎭 Выберите звуковой пресет / эффект:", reply_markup=get_effect_keyboard())

        @dp.callback_query(F.data == "open_effects")
        async def on_open_effects(callback: types.CallbackQuery):
            await callback.message.edit_text("🎭 Выберите звуковой пресет / эффект:", reply_markup=get_effect_keyboard())

        @dp.callback_query(F.data.startswith("effect_"))
        async def on_effect_selected(callback: types.CallbackQuery):
            user_id = callback.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                await callback.answer("Нет доступа", show_alert=True)
                return

            selected = callback.data.split("_")[1]
            config["effect"] = selected
            save_config(config)
            eff_name = dict(available_effects).get(selected, selected)
            await callback.message.edit_text(
                f"✅ Эффект изменен на: <b>{eff_name}</b>",
                parse_mode="HTML",
                reply_markup=get_effect_keyboard()
            )
            await callback.answer(f"Эффект: {eff_name}")

        @dp.message(Command("status"))
        async def cmd_status(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                return
            st = status_state["status"]
            q = tts_queue.qsize()
            eff_name = dict(available_effects).get(config.get("effect", "none"), "✨ Обычный")
            mon_st = "ВКЛ" if config.get("monitor_enabled") else "ВЫКЛ"
            spk_name = all_speakers_dict.get(config.get("speaker"), config.get("speaker"))
            tts_vol = int(config.get("volume", 1.0) * 100)
            sfx_vol = int(config.get("sfx_volume", 0.5) * 100)
            await msg.answer(
                f"📊 <b>Статус моста:</b>\n"
                f"• Состояние: <code>{st}</code>\n"
                f"• Голос: <b>{spk_name}</b>\n"
                f"• Эффект: <b>{eff_name}</b>\n"
                f"• Мониторинг: <b>{mon_st}</b>\n"
                f"• В очереди: <b>{q}</b> фраз\n"
                f"• Громкость речи: <b>{tts_vol}%</b>\n"
                f"• Громкость SFX (звуков): <b>{sfx_vol}%</b>",
                parse_mode="HTML"
            )

        @dp.message(F.audio | F.voice | F.document)
        async def on_audio_message(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                return

            file_obj = msg.audio or msg.voice or msg.document
            if not file_obj:
                return

            if msg.document:
                fn = (msg.document.file_name or "").lower()
                valid_exts = (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus")
                if not any(fn.endswith(ext) for ext in valid_exts):
                    return

            try:
                status_msg = await msg.reply("⏳ Загрузка и декодирование аудиофайла...")
                f_info = await bot.get_file(file_obj.file_id)
                import io
                dest = io.BytesIO()
                await bot.download_file(f_info.file_path, dest)
                dest.seek(0)
                audio_np = decode_audio_bytes(dest.read())
                if len(audio_np) == 0:
                    await status_msg.edit_text("❌ Не удалось прочитать или декодировать аудиофайл.")
                    return

                display_name = getattr(file_obj, "file_name", "Голосовое сообщение")
                enqueue_tts(display_name, raw_audio=audio_np)
                sfx_vol = int(config.get("sfx_volume", 0.5) * 100)
                await status_msg.edit_text(f"🔊 <b>Воспроизведение ({sfx_vol}%):</b> <code>{display_name}</code>", parse_mode="HTML")
                try:
                    await msg.react([types.ReactionTypeEmoji(emoji="🔥")])
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"[Telegram] Ошибка воспроизведения файла: {e}")
                await msg.reply(f"❌ Ошибка воспроизведения: {e}")

        @dp.message(F.text)
        async def on_text_message(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                return

            raw_text = msg.text.strip()
            chosen_spk = config.get("speaker")
            chosen_eff = config.get("effect", "none")
            text_to_speak = raw_text

            if raw_text.startswith("[") and "]" in raw_text:
                tag = raw_text[1:raw_text.index("]")].strip().lower()
                text_to_speak = raw_text[raw_text.index("]") + 1:].strip()
                valid_speakers = ["aidar", "baya", "kseniya", "xenia", "eugene", "random", "dmitri", "irina", "edge_dmitry", "edge", "dmitry_neural"]
                valid_effects = ["none", "radio", "chipmunk", "demon", "robot", "echo"]
                for p in tag.split(":"):
                    p = p.strip()
                    if p in valid_speakers:
                        chosen_spk = p
                    elif p in valid_effects:
                        chosen_eff = p

            if not text_to_speak:
                return

            enqueue_tts(text_to_speak, chosen_spk, chosen_eff)
            try:
                await msg.react([types.ReactionTypeEmoji(emoji="📢")])
            except Exception:
                pass

        @dp.message(Command("discord"))
        async def cmd_discord(msg: types.Message):
            user_id = msg.from_user.id
            configured_id = int(config.get("tg_user_id") or 0)
            if configured_id != 0 and user_id != configured_id:
                return

            token = config.get("discord_token", "").strip()
            if not token:
                await msg.answer("❌ <b>Discord-бот не настроен.</b>\nУкажите Discord Bot Token в веб-пульте на ПК (иконка 🤖 в шапке).", parse_mode="HTML")
                return

            online = discord_bot_instance is not None and discord_bot_instance.is_ready()
            vc = get_active_voice_client()
            in_voice = vc is not None and vc.is_connected()
            bot_name = str(discord_bot_instance.user) if online else "Не в сети"
            st_icon = "🟢" if online else "🟡"
            ch_str = f"🔊 <b>{vc.channel.name}</b> (сервер {vc.channel.guild.name})" if in_voice else "<i>Не подключен к каналу</i>"

            builder = InlineKeyboardBuilder()
            if online:
                for guild in discord_bot_instance.guilds:
                    for ch in guild.voice_channels[:6]:
                        builder.button(text=f"📞 {ch.name[:16]}", callback_data=f"djoin_{ch.id}")
                if in_voice:
                    builder.button(text="🚪 Выйти из канала", callback_data="dleave")
                builder.adjust(2)

            text = (
                f"🤖 <b>Discord Voicer Bot:</b>\n\n"
                f"• Имя: <b>{bot_name}</b>\n"
                f"• Статус: {st_icon} {'В сети' if online else 'Подключение...'}\n"
                f"• Голосовой канал: {ch_str}\n\n"
                f"💡 <i>Нажмите кнопку ниже, чтобы бот зашел в голосовой канал на сервере:</i>"
            )
            await msg.answer(text, parse_mode="HTML", reply_markup=builder.as_markup())

        @dp.callback_query(F.data.startswith("djoin_"))
        async def on_djoin_callback(callback: types.CallbackQuery):
            channel_id = int(callback.data.split("_")[1])
            if not discord_bot_instance or not discord_bot_instance.is_ready():
                await callback.answer("Бот не в сети", show_alert=True)
                return
            ch = discord_bot_instance.get_channel(channel_id)
            if not ch:
                try:
                    ch = await discord_bot_instance.fetch_channel(channel_id)
                except Exception:
                    ch = None
            if not ch or not isinstance(ch, discord.VoiceChannel):
                await callback.answer("Канал не найден", show_alert=True)
                return
            global active_discord_voice_client
            try:
                if ch.guild.voice_client:
                    vc = await ch.guild.voice_client.move_to(ch)
                else:
                    vc = await ch.connect(self_deaf=False, self_mute=False)
                active_discord_voice_client = vc
                config["discord_voice_channel_id"] = ch.id
                save_config(config)
                await callback.answer(f"Подключен к {ch.name}!")
                await callback.message.edit_text(
                    f"✅ Discord-бот успешно подключен к каналу: <b>{ch.name}</b> ({ch.guild.name})",
                    parse_mode="HTML"
                )
            except Exception as e:
                await callback.answer(f"Ошибка: {e}", show_alert=True)

        @dp.callback_query(F.data == "dleave")
        async def on_dleave_callback(callback: types.CallbackQuery):
            vc = get_active_voice_client()
            if vc and vc.is_connected():
                await vc.disconnect()
                config["discord_voice_channel_id"] = None
                save_config(config)
                await callback.answer("Отключен от канала")
                await callback.message.edit_text("👋 Discord-бот отключен от голосового канала.")
            else:
                await callback.answer("Не в канале")

        @dp.message(Command("dleave"))
        async def cmd_dleave(msg: types.Message):
            vc = get_active_voice_client()
            if vc and vc.is_connected():
                await vc.disconnect()
                config["discord_voice_channel_id"] = None
                save_config(config)
                await msg.answer("👋 Discord-бот отключен от голосового канала.")
            else:
                await msg.answer("Бот не находится в голосовом канале.")

        logger.info("[Telegram] Бот успешно авторизован и запущен!")
        await dp.start_polling(bot, handle_signals=False)
    except asyncio.CancelledError:
        logger.info("[Telegram] Опрос бота остановлен.")
    except Exception as e:
        logger.error(f"[Telegram] Ошибка запуска бота: {e}")


async def restart_telegram_bot():
    global tg_bot_task, tg_bot_instance
    if tg_bot_instance:
        try:
            await tg_bot_instance.session.close()
        except Exception:
            pass
    if tg_bot_task and not tg_bot_task.done():
        tg_bot_task.cancel()
        try:
            await tg_bot_task
        except asyncio.CancelledError:
            pass
    tg_bot_task = asyncio.create_task(start_telegram_bot_service())


# --- DISCORD BOT ИНТЕГРАЦИЯ (DISCORD.PY) ---
discord_bot_task: Optional[asyncio.Task] = None
discord_bot_instance: Optional[commands.Bot] = None


async def start_discord_bot_service():
    global discord_bot_instance, active_discord_voice_client
    token = config.get("discord_token", "").strip()
    if not token:
        logger.info("[Discord Bot] Токен не указан. Discord-бот не запускается.")
        return

    try:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.voice_states = True

        bot = commands.Bot(command_prefix="!", intents=intents)
        discord_bot_instance = bot

        @bot.event
        async def on_ready():
            logger.info(f"[Discord Bot] Авторизован как {bot.user} (ID: {bot.user.id})")
            saved_channel_id = config.get("discord_voice_channel_id")
            if saved_channel_id and config.get("discord_auto_connect", True):
                try:
                    ch = bot.get_channel(int(saved_channel_id))
                    if ch and isinstance(ch, discord.VoiceChannel):
                        global active_discord_voice_client
                        active_discord_voice_client = await ch.connect(self_deaf=False, self_mute=False)
                        logger.info(f"[Discord Bot] Автоматически подключился к каналу: {ch.name}")
                except Exception as e:
                    logger.warning(f"[Discord Bot] Не удалось авто-подключиться к каналу: {e}")

        @bot.command(name="join")
        async def cmd_join(ctx):
            if not ctx.author.voice or not ctx.author.voice.channel:
                await ctx.send("❌ Вы не находитесь в голосовом канале!")
                return
            channel = ctx.author.voice.channel
            global active_discord_voice_client
            if ctx.voice_client:
                await ctx.voice_client.move_to(channel)
                active_discord_voice_client = ctx.voice_client
            else:
                active_discord_voice_client = await channel.connect(self_deaf=False, self_mute=False)
            config["discord_voice_channel_id"] = channel.id
            save_config(config)
            await ctx.send(f"🔊 Подключился к голосовому каналу: **{channel.name}**")

        @bot.command(name="leave", aliases=["dc", "disconnect"])
        async def cmd_leave(ctx):
            global active_discord_voice_client
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
                active_discord_voice_client = None
                config["discord_voice_channel_id"] = None
                save_config(config)
                await ctx.send("👋 Отключился от голосового канала.")
            else:
                await ctx.send("Бот не находится в голосовом канале.")

        @bot.command(name="status")
        async def cmd_status(ctx):
            vc = ctx.voice_client
            ch_name = vc.channel.name if vc and vc.channel else "Не подключен"
            spk = config.get("speaker", "aidar")
            eff = config.get("effect", "none")
            await ctx.send(
                f"📊 **Статус Voicer Bot:**\n"
                f"• Голосовой канал: **{ch_name}**\n"
                f"• Текущий голос: **{spk}**\n"
                f"• Эффект: **{eff}**\n"
                f"• В очереди: **{tts_queue.qsize()}**"
            )

        @bot.command(name="vhelp", aliases=["voicerhelp"])
        async def cmd_vhelp(ctx):
            await ctx.send(
                "🎙️ **Команды Discord-бота:**\n"
                "`!join` — позвать бота в свой голосовой канал\n"
                "`!leave` — отключить бота от голосового канала\n"
                "`!status` — статус бота и канала\n\n"
                "💡 Текст пишите на сайте или в Telegram-боте — он сразу прозвучит в голосовом канале!"
            )

        await bot.start(token)
    except asyncio.CancelledError:
        logger.info("[Discord Bot] Сервис Discord-бота остановлен.")
    except Exception as e:
        logger.error(f"[Discord Bot] Ошибка запуска бота: {e}")


async def restart_discord_bot():
    global discord_bot_task, discord_bot_instance
    if discord_bot_instance and not discord_bot_instance.is_closed():
        try:
            await discord_bot_instance.close()
        except Exception:
            pass
    if discord_bot_task and not discord_bot_task.done():
        discord_bot_task.cancel()
        try:
            await discord_bot_task
        except asyncio.CancelledError:
            pass
    discord_bot_task = asyncio.create_task(start_discord_bot_service())


# --- ВЕБ-СЕРВЕР (FASTAPI) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tg_bot_task, discord_bot_task
    ip = get_local_ip()
    print("\n" + "=" * 65)
    print(" 🚀 ЛОКАЛЬНЫЙ ПУЛЬТ НА КОМПЬЮТЕРЕ:  http://127.0.0.1:8000")
    print(f" 📱 ПУЛЬТ ДЛЯ ТЕЛЕФОНА (по Wi-Fi): http://{ip}:8000")
    print(" 🎙️ ВЫВОД ЗВУКА: CABLE Input (VB-Audio Virtual Cable)")
    print(" 🤖 TELEGRAM-БОТ: @orkdownbot (https://t.me/orkdownbot) - АКТИВИРОВАН!")
    print(" 🎮 DISCORD-БОТ: Интеграция с голосовыми каналами Discord")
    print(" ⚙️ Настройки Telegram, Discord и выбор устройств доступны в веб-интерфейсе!")
    print("=" * 65 + "\n")
    tg_bot_task = asyncio.create_task(start_telegram_bot_service())
    discord_bot_task = asyncio.create_task(start_discord_bot_service())
    yield
    if tg_bot_instance:
        try:
            await tg_bot_instance.session.close()
        except Exception:
            pass
    if tg_bot_task and not tg_bot_task.done():
        tg_bot_task.cancel()
    if discord_bot_instance and not discord_bot_instance.is_closed():
        try:
            await discord_bot_instance.close()
        except Exception:
            pass
    if discord_bot_task and not discord_bot_task.done():
        discord_bot_task.cancel()


app = FastAPI(title="Silero TTS Discord Bridge", lifespan=lifespan)


class SpeakRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    effect: Optional[str] = None
    device_id: Optional[int] = None


class ConfigUpdateRequest(BaseModel):
    speaker: Optional[str] = None
    effect: Optional[str] = None
    cable_device_id: Optional[int] = None
    monitor_device_id: Optional[int] = None
    monitor_enabled: Optional[bool] = None
    output_cable_enabled: Optional[bool] = None
    output_discord_enabled: Optional[bool] = None
    volume: Optional[float] = None
    sfx_volume: Optional[float] = None
    quick_phrases: Optional[List[str]] = None


class TelegramUpdateRequest(BaseModel):
    tg_token: str
    tg_user_id: int


class DiscordTokenRequest(BaseModel):
    token: str


class DiscordJoinRequest(BaseModel):
    channel_id: Union[int, str]


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# --- HTML / JS / CSS ИНТЕРФЕЙС В СТИЛЕ DISCORD DARK THEME ---
HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Silero TTS Discord Bridge</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #111214;
            --bg-card: #1e1f22;
            --bg-input: #2b2d31;
            --bg-hover: #35373c;
            --accent: #5865f2;
            --accent-hover: #4752c4;
            --accent-green: #23a55a;
            --accent-green-hover: #1f914f;
            --accent-red: #da373c;
            --text-main: #f2f3f5;
            --text-muted: #949ba4;
            --text-faint: #6d6f78;
            --border: #3b3e45;
            --radius-lg: 16px;
            --radius-md: 10px;
            --radius-sm: 6px;
            --shadow: 0 8px 24px rgba(0,0,0,0.4);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, sans-serif; }
        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 20px 14px 40px;
        }
        .container {
            width: 100%;
            max-width: 680px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card);
            padding: 16px 20px;
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }
        .header-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 19px;
            font-weight: 700;
        }
        .header-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .icon-btn {
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text-main);
            width: 38px;
            height: 38px;
            border-radius: var(--radius-md);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.2s;
        }
        .icon-btn:hover { background: var(--bg-hover); transform: translateY(-1px); }

        /* Карточка статуса */
        .status-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 14px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--shadow);
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 500;
        }
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
        }
        .dot.speaking {
            background: #f0b232;
            box-shadow: 0 0 12px #f0b232;
            animation: pulse 1s infinite;
        }
        @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; } 100% { opacity: 0.5; } }
        .queue-badge {
            background: var(--bg-input);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            color: var(--text-muted);
            border: 1px solid var(--border);
        }

        /* Главная карточка ввода */
        .main-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 22px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .section-label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .voice-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        @media (max-width: 520px) {
            .voice-grid { grid-template-columns: repeat(2, 1fr); }
        }
        .voice-btn {
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 10px;
            border-radius: var(--radius-md);
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .voice-btn:hover { background: var(--bg-hover); }
        .voice-btn.active {
            border-color: var(--accent);
            background: rgba(88, 101, 242, 0.18);
        }
        .voice-name { font-size: 14px; font-weight: 600; }
        .voice-desc { font-size: 11px; color: var(--text-muted); }

        .textarea-wrapper {
            position: relative;
        }
        textarea {
            width: 100%;
            height: 95px;
            background: var(--bg-input);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 14px;
            font-size: 16px;
            resize: none;
            transition: border-color 0.2s;
            line-height: 1.4;
        }
        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(88, 101, 242, 0.3);
        }
        .textarea-hints {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-faint);
            margin-top: 4px;
            padding: 0 4px;
        }

        .action-row {
            display: flex;
            gap: 12px;
        }
        .btn-speak {
            flex: 1;
            background: var(--accent-green);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            padding: 14px 20px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 4px 12px rgba(35, 165, 90, 0.3);
        }
        .btn-speak:hover { background: var(--accent-green-hover); transform: translateY(-1px); }
        .btn-speak:active { transform: translateY(1px); }

        /* Зона Drag & Drop для аудиофайлов */
        .drop-zone {
            background: var(--bg-input);
            border: 2px dashed var(--border);
            border-radius: var(--radius-md);
            padding: 16px 14px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: var(--accent);
            background: rgba(88, 101, 242, 0.14);
        }

        /* Маршрутизация звука */
        .audio-section {
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .audio-row {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        select {
            width: 100%;
            background: var(--bg-card);
            color: var(--text-main);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 10px 12px;
            font-size: 13px;
            cursor: pointer;
            outline: none;
        }
        select:focus { border-color: var(--accent); }
        .toggle-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 13px;
        }
        .switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background-color: var(--bg-hover);
            transition: .3s;
            border-radius: 24px;
            border: 1px solid var(--border);
        }
        .slider:before {
            position: absolute; content: ""; height: 16px; width: 16px; left: 3px; bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }
        input:checked + .slider { background-color: var(--accent-green); }
        input:checked + .slider:before { transform: translateX(20px); }

        .volume-control {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 4px;
        }
        .volume-slider {
            flex: 1;
            accent-color: var(--accent);
            cursor: pointer;
        }

        /* Быстрые фразы (Саундборд) */
        .soundboard-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 20px;
            box-shadow: var(--shadow);
        }
        .quick-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 10px;
        }
        @media (max-width: 520px) {
            .quick-grid { grid-template-columns: 1fr; }
        }
        .quick-btn-wrapper {
            position: relative;
            display: flex;
        }
        .quick-btn {
            flex: 1;
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 10px 14px;
            border-radius: var(--radius-md);
            font-size: 13px;
            font-weight: 500;
            text-align: left;
            cursor: pointer;
            transition: all 0.15s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .quick-btn:hover {
            background: var(--bg-hover);
            border-color: var(--accent);
        }
        .quick-btn:active { transform: scale(0.98); }
        .quick-del {
            background: transparent;
            border: none;
            color: var(--text-faint);
            padding: 0 10px;
            cursor: pointer;
            font-size: 14px;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            transition: color 0.2s;
        }
        .quick-del:hover { color: var(--accent-red); }

        .add-phrase-row {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }
        .add-input {
            flex: 1;
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-radius: var(--radius-sm);
            padding: 8px 12px;
            font-size: 13px;
            outline: none;
        }
        .btn-add {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: var(--radius-sm);
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-add:hover { background: var(--accent-hover); }

        /* Модальные окна */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(4px);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .modal {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            width: 100%;
            max-width: 480px;
            padding: 24px;
            box-shadow: 0 12px 36px rgba(0,0,0,0.6);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 18px;
            font-weight: 700;
        }
        .modal-close {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 20px;
            cursor: pointer;
        }
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .input-group label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
        }
        .input-group input {
            background: var(--bg-input);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-radius: var(--radius-sm);
            padding: 10px 12px;
            font-size: 14px;
            outline: none;
        }
        .input-group input:focus { border-color: var(--accent); }
        .input-hint {
            font-size: 11px;
            color: var(--text-faint);
            line-height: 1.4;
        }
        .input-hint a { color: var(--accent); text-decoration: none; }
        .input-hint a:hover { text-decoration: underline; }
        .modal-btn {
            background: var(--accent);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            padding: 12px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 8px;
        }
        .modal-btn:hover { background: var(--accent-hover); }
        .qr-box {
            text-align: center;
            padding: 14px;
            background: var(--bg-input);
            border-radius: var(--radius-md);
            font-size: 14px;
            font-weight: 600;
            user-select: all;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Шапка -->
        <div class="header">
            <div class="header-title">
                <span>🎙️</span>
                <span>Silero TTS Bridge</span>
            </div>
            <div class="header-actions">
                <button class="icon-btn" title="Настройки Discord Бота (Голосовой канал)" onclick="openModal('discordModal')">🎮</button>
                <button class="icon-btn" title="Шпаргалка интонаций и ударений" onclick="openModal('cheatModal')">📖</button>
                <button class="icon-btn" title="QR-код / Телефон" onclick="openModal('phoneModal')">📱</button>
                <button class="icon-btn" title="Настройки Telegram" onclick="openModal('tgModal')">⚙️</button>
            </div>
        </div>

        <!-- Статус -->
        <div class="status-card">
            <div class="status-indicator">
                <div class="dot" id="statusDot"></div>
                <span id="statusText">Готов к работе</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <div class="queue-badge" id="discordBadge" onclick="openModal('discordModal')" style="cursor:pointer; background:rgba(88,101,242,0.15); border:1px solid rgba(88,101,242,0.3); color:#8590f5;" title="Клик для настройки Discord">🎮 Discord: Офлайн</div>
                <div class="queue-badge" id="queueBadge">Очередь: 0</div>
            </div>
        </div>

        <!-- Основная панель -->
        <div class="main-card">
            <div>
                <div class="section-label">
                    <span>🗣️ Silero TTS (Эмоциональные)</span>
                    <span style="font-size:11px; font-weight:normal; text-transform:none;">(Ударения знаком +)</span>
                </div>
                <div class="voice-grid">
                    <button class="voice-btn" data-speaker="aidar" onclick="selectSpeaker('aidar')">
                        <span class="voice-name">⚡ Айдар</span>
                        <span class="voice-desc">Мужской, бодрый</span>
                    </button>
                    <button class="voice-btn" data-speaker="eugene" onclick="selectSpeaker('eugene')">
                        <span class="voice-name">🎙️ Евгений</span>
                        <span class="voice-desc">Мужской, баритон</span>
                    </button>
                    <button class="voice-btn" data-speaker="xenia" onclick="selectSpeaker('xenia')">
                        <span class="voice-name">✨ Ксения</span>
                        <span class="voice-desc">Женский, четкий</span>
                    </button>
                    <button class="voice-btn" data-speaker="baya" onclick="selectSpeaker('baya')">
                        <span class="voice-name">🌸 Байя</span>
                        <span class="voice-desc">Женский, мягкий</span>
                    </button>
                    <button class="voice-btn" data-speaker="kseniya" onclick="selectSpeaker('kseniya')">
                        <span class="voice-name">🎭 Ксения 2</span>
                        <span class="voice-desc">Женский, эмоц.</span>
                    </button>
                    <button class="voice-btn" data-speaker="random" onclick="selectSpeaker('random')">
                        <span class="voice-name">🎲 Рандом</span>
                        <span class="voice-desc">Случайный голос</span>
                    </button>
                </div>
            </div>

            <div style="margin-top: 4px;">
                <div class="section-label">
                    <span>🎤 Piper TTS (Студийные)</span>
                    <span style="font-size:11px; font-weight:normal; text-transform:none;">(Идеальные числа и студийная дикция)</span>
                </div>
                <div class="voice-grid" style="grid-template-columns: repeat(2, 1fr);">
                    <button class="voice-btn" data-speaker="dmitri" onclick="selectSpeaker('dmitri')">
                        <span class="voice-name">⚡ Дмитрий</span>
                        <span class="voice-desc">Мужской, студийный VITS</span>
                    </button>
                    <button class="voice-btn" data-speaker="irina" onclick="selectSpeaker('irina')">
                        <span class="voice-name">🌸 Ирина</span>
                        <span class="voice-desc">Женский, студийный VITS</span>
                    </button>
                </div>
            </div>

            <div style="margin-top: 4px;">
                <div class="section-label">
                    <span>🌐 Edge TTS (Microsoft Neural)</span>
                    <span style="font-size:11px; font-weight:normal; text-transform:none;">(Облачный нейро-диктор от Microsoft)</span>
                </div>
                <div class="voice-grid" style="grid-template-columns: 1fr;">
                    <button class="voice-btn" data-speaker="edge_dmitry" onclick="selectSpeaker('edge_dmitry')">
                        <span class="voice-name">🌐 Дмитрий Neural (ru-RU-DmitryNeural)</span>
                        <span class="voice-desc">Естественный дикторский голос Microsoft Edge (бесплатно, без лимитов)</span>
                    </button>
                </div>
            </div>

            <div style="margin-top: 4px;">
                <div class="section-label">
                    <span>Звуковой эффект / Тембр</span>
                    <span style="font-size:11px; font-weight:normal; text-transform:none;">(быстрые пресеты без задержки)</span>
                </div>
                <div class="voice-grid">
                    <button class="voice-btn effect-btn" data-effect="none" onclick="selectEffect('none')">
                        <span class="voice-name">✨ Обычный</span>
                        <span class="voice-desc">Чистый нейро-голос</span>
                    </button>
                    <button class="voice-btn effect-btn" data-effect="radio" onclick="selectEffect('radio')">
                        <span class="voice-name">📻 Рация</span>
                        <span class="voice-desc">Военный переговорник</span>
                    </button>
                    <button class="voice-btn effect-btn" data-effect="chipmunk" onclick="selectEffect('chipmunk')">
                        <span class="voice-name">🐿️ Бурундук</span>
                        <span class="voice-desc">Высокий питч / Гном</span>
                    </button>
                    <button class="voice-btn effect-btn" data-effect="demon" onclick="selectEffect('demon')">
                        <span class="voice-name">👹 Демон</span>
                        <span class="voice-desc">Низкий бас / Монстр</span>
                    </button>
                    <button class="voice-btn effect-btn" data-effect="robot" onclick="selectEffect('robot')">
                        <span class="voice-name">🤖 Робот</span>
                        <span class="voice-desc">Кибернетический</span>
                    </button>
                    <button class="voice-btn effect-btn" data-effect="echo" onclick="selectEffect('echo')">
                        <span class="voice-name">📢 Эхо</span>
                        <span class="voice-desc">Арена / Пространство</span>
                    </button>
                </div>
            </div>

            <div class="textarea-wrapper">
                <textarea id="ttsInput" placeholder="Напиши фразу и нажми Enter для озвучки в Discord..."></textarea>
                <div class="textarea-hints">
                    <span>Нажми <b>Enter</b> для отправки, <b>Shift+Enter</b> для новой строки | <a href="javascript:void(0)" onclick="openModal('cheatModal')" style="color:var(--accent); text-decoration:none; font-weight:600;">📖 Шпаргалка интонаций (+)</a></span>
                    <span id="charCount">0 символов</span>
                </div>
            </div>

            <div class="action-row">
                <button class="btn-speak" onclick="sendTTS()">
                    <span>🔊</span>
                    <span>Сказать в Discord (Enter)</span>
                </button>
            </div>

            <!-- Загрузка аудиофайлов и музыки в Discord -->
            <div class="drop-zone" id="audioDropZone" onclick="document.getElementById('audioFileInput').click()">
                <input type="file" id="audioFileInput" accept="audio/*,.mp3,.wav,.ogg,.flac,.m4a,.aac,.opus" style="display:none;" onchange="handleFileSelect(this.files)">
                <div style="font-size: 26px;">🎵</div>
                <div style="font-weight: 600; font-size: 14px; margin-top: 4px;">Перетащите сюда любой аудиофайл (.mp3, .wav, .ogg, .flac)</div>
                <div style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">или кликните для выбора — файл мгновенно заиграет в Discord!</div>
                <div id="uploadStatus" style="font-size: 12px; color: var(--accent-green); margin-top: 6px; font-weight: 600; display: none;"></div>
            </div>

            <!-- Аудио Маршрутизация -->
            <div class="audio-section">
                <div class="toggle-row">
                    <span style="font-weight: 500;">🎮 Вывод в голосовой канал Discord:</span>
                    <label class="switch">
                        <input type="checkbox" id="discordOutputToggle" onchange="updateAudioConfig()">
                        <span class="slider"></span>
                    </label>
                </div>

                <div class="toggle-row">
                    <span style="font-weight: 500;">🎙️ Вывод в Виртуальный Кабель (микрофон Windows):</span>
                    <label class="switch">
                        <input type="checkbox" id="cableOutputToggle" onchange="updateAudioConfig()">
                        <span class="slider"></span>
                    </label>
                </div>

                <div class="audio-row" id="cableDevRow">
                    <span class="section-label">🎙️ Устройство Virtual Cable (CABLE Input):</span>
                    <select id="cableSelect" onchange="updateAudioConfig()"></select>
                </div>

                <div class="toggle-row">
                    <span style="font-weight: 500;">🎧 Мониторинг в свои наушники (слышать себя):</span>
                    <label class="switch">
                        <input type="checkbox" id="monitorToggle" onchange="updateAudioConfig()">
                        <span class="slider"></span>
                    </label>
                </div>

                <div class="audio-row" id="monitorDevRow" style="display:none;">
                    <span class="section-label">Устройство мониторинга (Твои наушники):</span>
                    <select id="monitorSelect" onchange="updateAudioConfig()"></select>
                </div>

                <div class="audio-row">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
                        <span>Громкость синтеза речи (TTS):</span>
                        <span id="volLabel">100%</span>
                    </div>
                    <div class="volume-control">
                        <span>🔈</span>
                        <input type="range" class="volume-slider" id="volSlider" min="10" max="200" value="100" oninput="onVolChange(this.value)" onchange="updateAudioConfig()">
                        <span>🔊</span>
                    </div>
                </div>

                <div class="audio-row">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted);">
                        <span>Громкость звуковых файлов / музыки (SFX):</span>
                        <span id="sfxVolLabel">50%</span>
                    </div>
                    <div class="volume-control">
                        <span>🔈</span>
                        <input type="range" class="volume-slider" id="sfxVolSlider" min="5" max="200" value="50" oninput="onSfxVolChange(this.value)" onchange="updateAudioConfig()">
                        <span>🔊</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Саундборд быстрых фраз -->
        <div class="soundboard-card">
            <div class="section-label">
                <span>⚡ Быстрые фразы (Клик для мгновенной озвучки)</span>
            </div>
            <div class="quick-grid" id="quickGrid"></div>
            <div class="add-phrase-row">
                <input type="text" class="add-input" id="newPhraseInput" placeholder="Новая фраза..." onkeydown="if(event.key==='Enter') addPhrase()">
                <button class="btn-add" onclick="addPhrase()">+ Добавить</button>
            </div>
        </div>
    </div>

    <!-- Модальное окно Discord -->
    <div class="modal-overlay" id="discordModal">
        <div class="modal" style="max-width: 580px; max-height: 85vh; overflow-y: auto;">
            <div class="modal-header">
                <span>🎮 Настройка Discord Бота</span>
                <button class="modal-close" onclick="closeModal('discordModal')">×</button>
            </div>
            
            <div id="discordStatusCard" style="background:var(--bg-input); padding:12px 16px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-weight:600; font-size:14px;" id="discordModalBotName">Бот не авторизован</div>
                    <div style="font-size:12px; color:var(--text-muted);" id="discordModalVoiceStatus">Голосовой канал: Не подключен</div>
                </div>
                <div id="discordModalStatusBadge" style="font-size:12px; padding:4px 10px; border-radius:12px; background:var(--accent-red); color:#fff; font-weight:600;">Офлайн</div>
            </div>

            <!-- Управление голосовым каналом -->
            <div style="display:flex; flex-direction:column; gap:8px; margin-top:12px;">
                <label style="font-size:12px; font-weight:600; color:var(--text-muted); text-transform:uppercase;">Подключиться к голосовому каналу:</label>
                <select id="discordVoiceSelect" style="width:100%; padding:10px 12px; background:var(--bg-input); border:1px solid var(--border); border-radius:var(--radius-md); color:#fff; font-size:14px; outline:none;">
                    <option value="">Загрузка списка серверов и каналов...</option>
                </select>
                <div style="display:flex; gap:10px; margin-top:4px;">
                    <button class="modal-btn" style="flex:1;" onclick="joinDiscordVoice()">🔊 Войти в канал</button>
                    <button class="modal-btn" style="flex:1; background:var(--bg-hover); color:var(--text-muted);" onclick="leaveDiscordVoice()">👋 Отключить</button>
                </div>
            </div>

            <!-- Ввод токена бота -->
            <div class="input-group" style="margin-top:14px;">
                <label>Discord Bot Token:</label>
                <input type="password" id="discordTokenInput" placeholder="Вставьте токен вашего Discord-бота...">
                <div class="input-hint">
                    Токен хранится только локально в <code>config.json</code> на вашем ПК.
                </div>
            </div>
            <button class="modal-btn" onclick="saveDiscordToken()">💾 Сохранить токен и перезапустить бота</button>

            <!-- Инструкция по созданию бота -->
            <div style="background:var(--bg-input); padding:14px; border-radius:var(--radius-md); font-size:12px; line-height:1.6; color:var(--text-muted); margin-top:12px;">
                <div style="font-weight:700; color:#fff; margin-bottom:6px; font-size:13px;">📖 Как подключить Discord-бота (инструкция):</div>
                1. Откройте <a href="https://discord.com/developers/applications" target="_blank" style="color:var(--accent); font-weight:600; text-decoration:none;">Discord Developer Portal</a> и нажмите <b>New Application</b>.<br>
                2. Перейдите во вкладку <b>Bot</b>, нажмите <b>Reset Token</b>, скопируйте токен и вставьте в поле выше.<br>
                3. На этой же странице <b>Bot</b> прокрутите вниз до <b>Privileged Gateway Intents</b> и включите:
                   <b>Message Content Intent</b> и <b>Server Members Intent</b>.<br>
                4. Перейдите в <b>OAuth2 → URL Generator</b>:<br>
                   &nbsp;&nbsp;&nbsp;&nbsp;• <b>Scopes:</b> отметьте галочкой <code>bot</code><br>
                   &nbsp;&nbsp;&nbsp;&nbsp;• <b>Bot Permissions:</b> <code>Connect</code>, <code>Speak</code>, <code>Send Messages</code>, <code>View Channels</code><br>
                5. Скопируйте полученную ссылку внизу, откройте её в браузере и пригласите бота на свой сервер.<br>
                6. 💡 <b>Подключение в канал:</b> выберите канал из списка выше и нажмите <i>«Войти в канал»</i>, либо зайдите в канал в Discord и напишите команду <code>!join</code> в текстовом чате!
            </div>
        </div>
    </div>

    <!-- Модальное окно Telegram -->
    <div class="modal-overlay" id="tgModal">
        <div class="modal">
            <div class="modal-header">
                <span>⚙️ Настройки Telegram Бота</span>
                <button class="modal-close" onclick="closeModal('tgModal')">×</button>
            </div>
            <div class="input-group">
                <label>Telegram Bot Token:</label>
                <input type="password" id="tgTokenInput" placeholder="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ">
                <div class="input-hint">
                    Получить токен можно у официального бота <a href="https://t.me/BotFather" target="_blank">@BotFather</a> в Telegram (команда /newbot).
                </div>
            </div>
            <div class="input-group">
                <label>Твой Telegram User ID (только цифры):</label>
                <input type="number" id="tgUserIdInput" placeholder="123456789">
                <div class="input-hint">
                    Узнать свой ID можно через бота <a href="https://t.me/userinfobot" target="_blank">@userinfobot</a>. Защищает бота от посторонних. Оставь 0, если хочешь открыть доступ.
                </div>
            </div>
            <button class="modal-btn" onclick="saveTelegramSettings()">💾 Сохранить и применить</button>
        </div>
    </div>

    <!-- Модальное окно Подключения с телефона -->
    <div class="modal-overlay" id="phoneModal">
        <div class="modal">
            <div class="modal-header">
                <span>📱 Доступ со смартфона</span>
                <button class="modal-close" onclick="closeModal('phoneModal')">×</button>
            </div>
            <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">
                Убедись, что телефон подключен к тому же <b>Wi-Fi</b> роутеру, что и компьютер, и перейди по ссылке:
            </div>
            <div class="qr-box" id="lanUrlBox">Загрузка...</div>
            <button class="modal-btn" onclick="navigator.clipboard.writeText(document.getElementById('lanUrlBox').innerText); alert('Ссылка скопирована!');">📋 Скопировать ссылку</button>
        </div>
    </div>

    <!-- Модальное окно Шпаргалки интонаций и ударений -->
    <div class="modal-overlay" id="cheatModal">
        <div class="modal" style="max-width: 580px; max-height: 85vh; overflow-y: auto;">
            <div class="modal-header">
                <span>📖 Шпаргалка: Silero, Piper и Аудиофайлы</span>
                <button class="modal-close" onclick="closeModal('cheatModal')">×</button>
            </div>
            <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">
                Сравнение двух движков озвучки и примеры работы:
            </div>
            <div style="display:flex; flex-direction:column; gap:10px; margin-top:8px;">
                <!-- Секция Silero -->
                <div style="font-size: 13px; font-weight:700; color: #5865f2; text-transform:uppercase; margin-top:4px;">
                    🗣️ Движок Silero TTS (Эмоциональный)
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">+ перед гласной (Ударение)</div>
                        <div style="font-size:12px; color:var(--text-muted);">з+амок (крепость) или зам+ок (дверной)</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Впереди старинный з+амок, а на двери висит зам+ок', 'aidar'); closeModal('cheatModal');">Сказать</button>
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">... (Многоточие — пауза ~0.5с)</div>
                        <div style="font-size:12px; color:var(--text-muted);">Создает задумчивую паузу с вздохом</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Ну... даже не знаю... надо подумать...', 'aidar'); closeModal('cheatModal');">Сказать</button>
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">- или — (Тире — смысловой акцент)</div>
                        <div style="font-size:12px; color:var(--text-muted);">Короткая смысловая пауза со сменой тона</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Парни — это сто процентов победа!', 'aidar'); closeModal('cheatModal');">Сказать</button>
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">? и ! (Вопрос и Восклицание)</div>
                        <div style="font-size:12px; color:var(--text-muted);">Повышение высоты тона и эмоциональность</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Ты серьезно?! Да ладно, не верю!', 'aidar'); closeModal('cheatModal');">Сказать</button>
                </div>

                <!-- Секция Piper -->
                <div style="font-size: 13px; font-weight:700; color: #23a55a; text-transform:uppercase; margin-top:8px;">
                    🎤 Движок Piper TTS (Студийный диктор)
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">Студийная четкость VITS</div>
                        <div style="font-size:12px; color:var(--text-muted);">Обучен на фонемах, звучит без артефактов</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Здравствуйте! Это студийный голос Дмитрий, движок Piper.', 'dmitri'); closeModal('cheatModal');">Тест</button>
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">Авто-чтение любых чисел и дат</div>
                        <div style="font-size:12px; color:var(--text-muted);">Идеально читает числа, проценты, даты и дроби</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Сегодня 1 ноября 2026 года, онлайн 14850 человек, шанс 99.5%.', 'dmitri'); closeModal('cheatModal');">Тест</button>
                </div>

                <!-- Секция Edge TTS -->
                <div style="font-size: 13px; font-weight:700; color: #0078d4; text-transform:uppercase; margin-top:8px;">
                    🌐 Движок Edge TTS (Microsoft Neural)
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">ru-RU-DmitryNeural</div>
                        <div style="font-size:12px; color:var(--text-muted);">Естественный мужской дикторский голос Microsoft Edge</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Привет! Это естественный голос Дмитрий от Edge TTS.', 'edge_dmitry'); closeModal('cheatModal');">Тест</button>
                </div>

                <!-- Секция Музыки и Звуков -->
                <div style="font-size: 13px; font-weight:700; color: #f0b232; text-transform:uppercase; margin-top:8px;">
                    🎵 Аудиофайлы и Музыка
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md);">
                    <div style="font-weight:600; color:#fff;">Перетаскивание файлов</div>
                    <div style="font-size:12px; color:var(--text-muted); margin-top:3px; line-height:1.4;">
                        Вы можете перетащить любой файл (.mp3, .wav, .ogg, .flac) прямо на сайт в поле с нотой 🎵 или отправить аудио/голосовое сообщение в Telegram-бота — звук мгновенно пойдет в микрофон Discord!
                    </div>
                </div>

                <!-- Секция Эффектов -->
                <div style="font-size: 13px; font-weight:700; color: #eb459e; text-transform:uppercase; margin-top:8px;">
                    🎭 Звуковые эффекты и теги
                </div>
                <div style="background:var(--bg-input); padding:10px 14px; border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-weight:600; color:#fff;">Быстрые теги в тексте</div>
                        <div style="font-size:12px; color:var(--text-muted);">[radio], [demon], [chipmunk], [robot], [echo]</div>
                    </div>
                    <button class="btn-add" onclick="sendTTS('Прием, проверка армейской рации!', 'aidar', 'radio'); closeModal('cheatModal');">Тест</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSpeaker = "aidar";
        let currentEffect = "none";
        let quickPhrases = [];

        function openModal(id) { document.getElementById(id).style.display = 'flex'; }
        function closeModal(id) { document.getElementById(id).style.display = 'none'; }
        window.onclick = function(e) {
            if (e.target.classList.contains('modal-overlay')) {
                e.target.style.display = 'none';
            }
        };

        // Загрузка устройств и конфига
        async function init() {
            try {
                const res = await fetch('/api/config');
                const data = await res.json();
                
                currentSpeaker = data.speaker || "aidar";
                currentEffect = data.effect || "none";
                quickPhrases = data.quick_phrases || [];
                updateSpeakerButtons();
                updateEffectButtons();
                renderQuickPhrases();

                document.getElementById('monitorToggle').checked = !!data.monitor_enabled;
                document.getElementById('monitorDevRow').style.display = data.monitor_enabled ? 'flex' : 'none';
                
                document.getElementById('discordOutputToggle').checked = data.output_discord_enabled !== undefined ? !!data.output_discord_enabled : true;
                document.getElementById('cableOutputToggle').checked = data.output_cable_enabled !== undefined ? !!data.output_cable_enabled : true;
                document.getElementById('cableDevRow').style.display = (data.output_cable_enabled !== false) ? 'flex' : 'none';
                
                const vol = Math.round((data.volume || 1.0) * 100);
                document.getElementById('volSlider').value = vol;
                document.getElementById('volLabel').innerText = vol + '%';

                const sfxVol = Math.round((data.sfx_volume !== undefined ? data.sfx_volume : 0.5) * 100);
                document.getElementById('sfxVolSlider').value = sfxVol;
                document.getElementById('sfxVolLabel').innerText = sfxVol + '%';

                document.getElementById('tgTokenInput').value = data.tg_token || "";
                document.getElementById('tgUserIdInput').value = data.tg_user_id || 0;
                document.getElementById('lanUrlBox').innerText = data.lan_url;

                // Загрузка списков устройств
                const devRes = await fetch('/api/devices');
                const devData = await devRes.json();
                
                const cableSelect = document.getElementById('cableSelect');
                const monitorSelect = document.getElementById('monitorSelect');
                cableSelect.innerHTML = '';
                monitorSelect.innerHTML = '';

                devData.devices.forEach(d => {
                    const opt1 = new Option(d.name, d.id);
                    if (d.id === devData.current_cable) opt1.selected = true;
                    cableSelect.add(opt1);

                    const opt2 = new Option(d.name, d.id);
                    if (d.id === devData.current_monitor) opt2.selected = true;
                    monitorSelect.add(opt2);
                });

            } catch (err) {
                console.error("Ошибка инициализации:", err);
            }
        }

        function updateSpeakerButtons() {
            document.querySelectorAll('.voice-btn:not(.effect-btn)').forEach(btn => {
                if (btn.getAttribute('data-speaker') === currentSpeaker) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        async function selectSpeaker(spk) {
            currentSpeaker = spk;
            updateSpeakerButtons();
            await fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({speaker: spk})
            });
        }

        function updateEffectButtons() {
            document.querySelectorAll('.effect-btn').forEach(btn => {
                if (btn.getAttribute('data-effect') === currentEffect) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }

        async function selectEffect(eff) {
            currentEffect = eff;
            updateEffectButtons();
            await fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({effect: eff})
            });
        }

        function onVolChange(val) {
            document.getElementById('volLabel').innerText = val + '%';
        }

        function onSfxVolChange(val) {
            document.getElementById('sfxVolLabel').innerText = val + '%';
        }

        async function updateAudioConfig() {
            const cable_device_id = parseInt(document.getElementById('cableSelect').value);
            const monitor_enabled = document.getElementById('monitorToggle').checked;
            document.getElementById('monitorDevRow').style.display = monitor_enabled ? 'flex' : 'none';
            const monitor_device_id = parseInt(document.getElementById('monitorSelect').value);
            const output_discord_enabled = document.getElementById('discordOutputToggle').checked;
            const output_cable_enabled = document.getElementById('cableOutputToggle').checked;
            document.getElementById('cableDevRow').style.display = output_cable_enabled ? 'flex' : 'none';
            const volume = parseFloat(document.getElementById('volSlider').value) / 100.0;
            const sfx_volume = parseFloat(document.getElementById('sfxVolSlider').value) / 100.0;

            await fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    cable_device_id,
                    monitor_enabled,
                    monitor_device_id,
                    output_discord_enabled,
                    output_cable_enabled,
                    volume,
                    sfx_volume
                })
            });
        }

        async function sendTTS(customText, customSpeaker, customEffect) {
            const input = document.getElementById('ttsInput');
            const text = (customText !== undefined ? customText : input.value).trim();
            if (!text) return;

            if (customText === undefined) {
                input.value = '';
                document.getElementById('charCount').innerText = '0 символов';
            }

            try {
                await fetch('/api/speak', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        text: text,
                        speaker: customSpeaker || currentSpeaker,
                        effect: customEffect || currentEffect
                    })
                });
            } catch (e) {
                console.error("Ошибка отправки речи:", e);
            }
        }

        // Рендер фраз саундборда
        function renderQuickPhrases() {
            const grid = document.getElementById('quickGrid');
            grid.innerHTML = '';
            quickPhrases.forEach((phrase, idx) => {
                const wrap = document.createElement('div');
                wrap.className = 'quick-btn-wrapper';

                const btn = document.createElement('button');
                btn.className = 'quick-btn';
                btn.title = phrase;
                btn.innerText = phrase;
                btn.onclick = () => sendTTS(phrase);

                const del = document.createElement('button');
                del.className = 'quick-del';
                del.innerHTML = '✕';
                del.title = 'Удалить фразу';
                del.onclick = (e) => {
                    e.stopPropagation();
                    deletePhrase(idx);
                };

                wrap.appendChild(btn);
                wrap.appendChild(del);
                grid.appendChild(wrap);
            });
        }

        async function addPhrase() {
            const input = document.getElementById('newPhraseInput');
            const val = input.value.trim();
            if (!val) return;
            quickPhrases.push(val);
            input.value = '';
            renderQuickPhrases();
            await syncPhrases();
        }

        async function deletePhrase(idx) {
            quickPhrases.splice(idx, 1);
            renderQuickPhrases();
            await syncPhrases();
        }

        async function syncPhrases() {
            await fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({quick_phrases: quickPhrases})
            });
        }

        async function saveTelegramSettings() {
            const tg_token = document.getElementById('tgTokenInput').value.trim();
            const tg_user_id = parseInt(document.getElementById('tgUserIdInput').value) || 0;
            
            try {
                const res = await fetch('/api/telegram', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({tg_token, tg_user_id})
                });
                if (res.ok) {
                    alert('✅ Настройки Telegram сохранены! Бот запущен/перезапущен.');
                    closeModal('tgModal');
                } else {
                    alert('Ошибка сохранения настроек Telegram');
                }
            } catch (e) {
                alert('Сетевая ошибка сохранения');
            }
        }

        async function saveDiscordToken() {
            const token = document.getElementById('discordTokenInput').value.trim();
            if (!token) {
                alert('Пожалуйста, введите токен бота');
                return;
            }
            try {
                const res = await fetch('/api/discord/token', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({token})
                });
                if (res.ok) {
                    alert('✅ Токен Discord сохранён! Бот подключается...');
                    setTimeout(pollDiscordStatus, 1500);
                } else {
                    alert('Ошибка сохранения токена');
                }
            } catch (e) {
                alert('Сетевая ошибка: ' + e);
            }
        }

        async function joinDiscordVoice() {
            const select = document.getElementById('discordVoiceSelect');
            const chId = select.value ? select.value.trim() : "";
            if (!chId) {
                alert('Пожалуйста, выберите голосовой канал из списка');
                return;
            }
            try {
                const res = await fetch('/api/discord/join', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({channel_id: chId})
                });
                const data = await res.json();
                if (data.status === 'joined') {
                    alert(`✅ Бот подключился к каналу "${data.channel_name}"!`);
                    pollDiscordStatus();
                } else {
                    alert('❌ ' + (data.message || 'Ошибка подключения'));
                }
            } catch (e) {
                alert('Ошибка сети: ' + e);
            }
        }

        async function leaveDiscordVoice() {
            try {
                const res = await fetch('/api/discord/leave', {method: 'POST'});
                const data = await res.json();
                if (data.status === 'disconnected') {
                    alert('👋 Бот отключен от голосового канала');
                    pollDiscordStatus();
                }
            } catch (e) {
                alert('Ошибка сети: ' + e);
            }
        }

        let lastDiscordVoiceId = null;
        async function pollDiscordStatus() {
            try {
                const res = await fetch('/api/discord');
                const data = await res.json();

                const badge = document.getElementById('discordBadge');
                const modalBotName = document.getElementById('discordModalBotName');
                const modalVoice = document.getElementById('discordModalVoiceStatus');
                const modalBadge = document.getElementById('discordModalStatusBadge');
                const select = document.getElementById('discordVoiceSelect');

                if (data.is_ready) {
                    modalBotName.innerText = `🤖 ${data.bot_name || 'Discord Bot'}`;
                    if (data.voice_connected) {
                        badge.style.background = 'rgba(35,165,90,0.2)';
                        badge.style.border = '1px solid rgba(35,165,90,0.5)';
                        badge.style.color = '#57f287';
                        badge.innerText = `🔊 ${data.current_channel_name || 'В канале'}`;
                        modalVoice.innerText = `🔊 В канале: ${data.current_channel_name}`;
                        modalBadge.style.background = 'var(--accent-green)';
                        modalBadge.innerText = 'В канале';
                    } else {
                        badge.style.background = 'rgba(88,101,242,0.2)';
                        badge.style.border = '1px solid rgba(88,101,242,0.5)';
                        badge.style.color = '#8590f5';
                        badge.innerText = '🎮 Онлайн (не в канале)';
                        modalVoice.innerText = 'Голосовой канал: Не подключен';
                        modalBadge.style.background = 'var(--accent)';
                        modalBadge.innerText = 'Онлайн';
                    }

                    // Обновляем список голосовых каналов
                    if (String(data.current_channel_id || '') !== String(lastDiscordVoiceId || '') || select.options.length <= 1) {
                        lastDiscordVoiceId = data.current_channel_id;
                        select.innerHTML = '<option value="">-- Выберите голосовой канал --</option>';
                        if (data.guilds && data.guilds.length > 0) {
                            data.guilds.forEach(g => {
                                const optgroup = document.createElement('optgroup');
                                optgroup.label = `📁 ${g.name}`;
                                g.channels.forEach(ch => {
                                    const opt = document.createElement('option');
                                    opt.value = ch.id;
                                    opt.innerText = `🔊 ${ch.name}`;
                                    if (String(ch.id) === String(data.current_channel_id)) {
                                        opt.selected = true;
                                    }
                                    optgroup.appendChild(opt);
                                });
                                select.appendChild(optgroup);
                            });
                        } else {
                            select.innerHTML = '<option value="">Бот ещё не добавлен на сервер</option>';
                        }
                    }
                } else if (data.token_set) {
                    badge.style.background = 'rgba(240,178,50,0.2)';
                    badge.style.border = '1px solid rgba(240,178,50,0.5)';
                    badge.style.color = '#f0b232';
                    badge.innerText = '🎮 Подключение бота...';
                    modalBotName.innerText = 'Подключение к Discord...';
                    modalBadge.style.background = '#f0b232';
                    modalBadge.innerText = 'Подключение';
                } else {
                    badge.style.background = 'rgba(100,100,100,0.2)';
                    badge.style.border = '1px solid rgba(100,100,100,0.4)';
                    badge.style.color = 'var(--text-muted)';
                    badge.innerText = '🎮 Discord: Офлайн';
                    modalBotName.innerText = 'Токен не указан';
                    modalBadge.style.background = 'var(--accent-red)';
                    modalBadge.innerText = 'Офлайн';
                }
            } catch (e) {}
            setTimeout(pollDiscordStatus, 3000);
        }

        // Горячая клавиша Enter в textarea
        const ttsInput = document.getElementById('ttsInput');
        ttsInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendTTS();
            }
        });
        ttsInput.addEventListener('input', function() {
            document.getElementById('charCount').innerText = `${this.value.length} символов`;
        });

        // Обработка Drag & Drop аудиофайлов
        const dropZone = document.getElementById('audioDropZone');
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('dragover');
            }, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('dragover');
            }, false);
        });
        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFileSelect(files);
        }, false);

        async function handleFileSelect(files) {
            if (!files || files.length === 0) return;
            const file = files[0];
            const status = document.getElementById('uploadStatus');
            status.style.display = 'block';
            status.style.color = '#f0b232';
            status.innerText = `⏳ Отправка "${file.name}" в Discord...`;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/play_audio_file', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.status === 'enqueued') {
                    status.style.color = 'var(--accent-green)';
                    status.innerText = `✅ Играет в Discord: ${data.filename}`;
                } else {
                    status.style.color = 'var(--accent-red)';
                    status.innerText = `❌ Ошибка: ${data.message || 'Сбой'}`;
                }
            } catch (e) {
                status.style.color = 'var(--accent-red)';
                status.innerText = `❌ Ошибка сети: ${e}`;
            }
            setTimeout(() => {
                status.style.display = 'none';
            }, 5000);
        }

        // Поллинг статуса
        async function pollStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                const dot = document.getElementById('statusDot');
                const text = document.getElementById('statusText');
                const badge = document.getElementById('queueBadge');

                badge.innerText = `Очередь: ${data.queue_size}`;

                if (data.status === 'speaking') {
                    dot.className = 'dot speaking';
                    text.innerText = `Озвучивание: "${data.text.substring(0, 30)}${data.text.length > 30 ? '...' : ''}"`;
                } else if (data.status === 'generating') {
                    dot.className = 'dot speaking';
                    text.innerText = 'Генерация нейросетью...';
                } else {
                    dot.className = 'dot';
                    text.innerText = 'Готов к работе';
                }
            } catch (e) {}
            setTimeout(pollStatus, 800);
        }

        init();
        pollStatus();
        pollDiscordStatus();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTML_PAGE


@app.get("/api/status")
async def get_status():
    vc = get_active_voice_client()
    return {
        "status": status_state["status"],
        "text": status_state["current_text"],
        "speaker": status_state["speaker"],
        "queue_size": tts_queue.qsize(),
        "tg_active": tg_bot_task is not None and not tg_bot_task.done(),
        "discord_ready": discord_bot_instance is not None and discord_bot_instance.is_ready(),
        "discord_voice": vc is not None and vc.is_connected(),
        "discord_channel": vc.channel.name if vc and vc.is_connected() and vc.channel else None
    }


@app.get("/api/devices")
async def get_devices():
    devices = []
    try:
        for idx, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) > 0:
                devices.append({
                    "id": idx,
                    "name": f"[{idx}] {d['name']} ({d.get('hostapi', '')})"
                })
    except Exception as e:
        logger.debug(f"Не удалось получить список звуковых устройств: {e}")
    return {
        "devices": devices,
        "current_cable": config.get("cable_device_id"),
        "current_monitor": config.get("monitor_device_id")
    }


@app.get("/api/config")
async def get_config():
    ip = get_local_ip()
    return {
        "speaker": config.get("speaker", "aidar"),
        "effect": config.get("effect", "none"),
        "cable_device_id": config.get("cable_device_id"),
        "monitor_device_id": config.get("monitor_device_id"),
        "monitor_enabled": config.get("monitor_enabled", True),
        "output_cable_enabled": config.get("output_cable_enabled", True),
        "output_discord_enabled": config.get("output_discord_enabled", True),
        "volume": config.get("volume", 1.0),
        "sfx_volume": config.get("sfx_volume", 0.5),
        "quick_phrases": config.get("quick_phrases", []),
        "tg_token": config.get("tg_token", ""),
        "tg_user_id": config.get("tg_user_id", 0),
        "lan_url": f"http://{ip}:8000"
    }


@app.post("/api/speak")
async def api_speak(req: SpeakRequest):
    spk = req.speaker or config.get("speaker", "aidar")
    eff = req.effect or config.get("effect", "none")
    if req.device_id is not None:
        config["cable_device_id"] = req.device_id
    enqueue_tts(req.text, spk, eff)
    return {"status": "enqueued", "queue_size": tts_queue.qsize()}


@app.post("/api/config")
async def update_config(req: ConfigUpdateRequest):
    if req.speaker is not None:
        config["speaker"] = req.speaker
    if req.effect is not None:
        config["effect"] = req.effect
    if req.cable_device_id is not None:
        config["cable_device_id"] = req.cable_device_id
    if req.monitor_device_id is not None:
        config["monitor_device_id"] = req.monitor_device_id
    if req.monitor_enabled is not None:
        config["monitor_enabled"] = req.monitor_enabled
    if req.output_cable_enabled is not None:
        config["output_cable_enabled"] = req.output_cable_enabled
    if req.output_discord_enabled is not None:
        config["output_discord_enabled"] = req.output_discord_enabled
    if req.volume is not None:
        config["volume"] = req.volume
    if req.sfx_volume is not None:
        config["sfx_volume"] = req.sfx_volume
    if req.quick_phrases is not None:
        config["quick_phrases"] = req.quick_phrases
    save_config(config)
    return {"status": "updated"}


@app.post("/api/telegram")
async def update_telegram(req: TelegramUpdateRequest):
    config["tg_token"] = req.tg_token.strip()
    config["tg_user_id"] = req.tg_user_id
    save_config(config)
    await restart_telegram_bot()
    return {"status": "restarted"}


@app.get("/api/discord")
async def get_discord_info():
    is_ready = discord_bot_instance is not None and discord_bot_instance.is_ready()
    bot_name = str(discord_bot_instance.user) if is_ready and discord_bot_instance.user else None
    token_set = bool(config.get("discord_token", "").strip())
    
    vc = get_active_voice_client()
    voice_connected = vc is not None and vc.is_connected()
    current_channel_id = str(vc.channel.id) if voice_connected and vc.channel else None
    current_channel_name = vc.channel.name if voice_connected and vc.channel else None
    
    guilds_data = []
    if is_ready:
        for g in discord_bot_instance.guilds:
            v_channels = []
            for ch in g.voice_channels:
                v_channels.append({"id": str(ch.id), "name": ch.name})
            guilds_data.append({"id": str(g.id), "name": g.name, "channels": v_channels})
            
    return {
        "is_ready": is_ready,
        "bot_name": bot_name,
        "token_set": token_set,
        "voice_connected": voice_connected,
        "current_channel_id": current_channel_id,
        "current_channel_name": current_channel_name,
        "guilds": guilds_data,
        "output_discord_enabled": config.get("output_discord_enabled", True),
        "output_cable_enabled": config.get("output_cable_enabled", True)
    }


@app.post("/api/discord/token")
async def update_discord_token(req: DiscordTokenRequest):
    config["discord_token"] = req.token.strip()
    save_config(config)
    await restart_discord_bot()
    return {"status": "restarted"}


@app.post("/api/discord/join")
async def discord_join_channel(req: DiscordJoinRequest):
    global active_discord_voice_client
    if not discord_bot_instance or not discord_bot_instance.is_ready():
        return {"status": "error", "message": "Discord-бот не авторизован или выключен. Проверьте токен в настройках Discord."}
    
    try:
        ch_id = int(str(req.channel_id).strip())
    except (ValueError, TypeError):
        return {"status": "error", "message": "Некорректный ID голосового канала."}

    ch = discord_bot_instance.get_channel(ch_id)
    if ch is None:
        try:
            ch = await discord_bot_instance.fetch_channel(ch_id)
        except Exception as e:
            logger.warning(f"[Discord Bot] fetch_channel({ch_id}) не удался: {e}")
            ch = None

    if not ch or not isinstance(ch, discord.VoiceChannel):
        return {"status": "error", "message": f"Голосовой канал ({ch_id}) не найден или недоступен боту."}
    
    try:
        if ch.guild.voice_client:
            vc = ch.guild.voice_client
            if vc.channel.id != ch.id:
                await vc.move_to(ch)
            active_discord_voice_client = vc
        else:
            active_discord_voice_client = await ch.connect(self_deaf=False, self_mute=False)

        config["discord_voice_channel_id"] = ch.id
        save_config(config)
        logger.info(f"[Discord Bot] Успешно подключен к каналу: {ch.name} (сервер {ch.guild.name})")
        return {"status": "joined", "channel_name": ch.name}
    except Exception as e:
        logger.error(f"[Discord Bot] Ошибка подключения к голосовому каналу: {e}")
        return {"status": "error", "message": f"Ошибка подключения к каналу: {e}"}


@app.post("/api/discord/leave")
async def discord_leave_channel():
    global active_discord_voice_client
    vc = get_active_voice_client()
    if vc:
        try:
            await vc.disconnect()
        except Exception:
            pass
        active_discord_voice_client = None
    config["discord_voice_channel_id"] = None
    save_config(config)
    return {"status": "disconnected"}


@app.post("/api/play_audio_file")
async def api_play_audio_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        audio_data = decode_audio_bytes(content)
        if len(audio_data) == 0:
            return {"status": "error", "message": "Пустой или неподдерживаемый аудиофайл"}
        display_name = file.filename or "Аудиофайл"
        enqueue_tts(display_name, raw_audio=audio_data)
        return {
            "status": "enqueued",
            "filename": display_name,
            "queue_size": tts_queue.qsize()
        }
    except Exception as e:
        logger.error(f"Ошибка воспроизведения аудиофайла {file.filename}: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
