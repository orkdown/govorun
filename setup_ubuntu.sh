#!/bin/bash
set -e

echo "=== [1/5] Обновление пакетов и установка системных зависимостей Ubuntu ==="
sudo apt update
sudo apt install -y python3 python3-pip python3-venv ffmpeg git libopus0

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

echo "=== [2/5] Создание виртуального окружения Python ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "=== [3/5] Установка Python-зависимостей ==="
source venv/bin/activate
pip install --upgrade pip

# Установка CPU-версии PyTorch (весит ~150МБ вместо 2ГБ CUDA)
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

echo "=== [4/5] Подготовка конфигурации ==="
if [ ! -f "config.json" ]; then
    cp config.example.json config.json
    echo "[!] Создан config.json из шаблона. Не забудьте вписать туда ваши токены!"
fi

echo "=== [5/5] Создание systemd-сервиса (автозапуск 24/7) ==="
SERVICE_NAME="voicer"
CURRENT_USER="$(whoami)"

sudo bash -c "cat > /etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=Silero TTS Web & Discord Bridge
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/venv/bin/python ${APP_DIR}/app.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

echo "============================================================"
echo " [OK] Установка завершена!"
echo " Для запуска выполните: sudo systemctl start ${SERVICE_NAME}"
echo " Проверить статус:       sudo systemctl status ${SERVICE_NAME}"
echo " Просмотр логов:        sudo journalctl -u ${SERVICE_NAME} -f"
echo " Веб-панель доступна:   http://$(curl -s ifconfig.me || hostname -I | awk '{print \$1}'):8000"
echo "============================================================"
