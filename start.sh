#!/usr/bin/env bash
# start.sh - start script for FAHAD bot (venv-enabled)
# This version creates a temporary virtualenv to run pip as non-root and avoid pip warnings.
# If AUTO_DOWNLOAD=true it will also install transformers/torch in the venv before starting the bot
# so the model generation from transformers can be used (may take time and bandwidth).

set -e

VENV_DIR="/tmp/fahad_venv"

echo "إعداد بيئة افتراضية (venv) في: ${VENV_DIR}"
# try to create venv; tolerate failure
python -m venv "${VENV_DIR}" || python3 -m venv "${VENV_DIR}" || true

# If venv exists, activate it
if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "${VENV_DIR}/bin/activate"
  echo "بيئة venv مفعلة"
  # upgrade pip inside venv (non-fatal)
  pip install --upgrade pip || true
else
  echo "فشل إنشاء/تفعيل venv؛ سيتم استخدام بايثون النظامي"
fi

echo "تثبيت الحزم الأساسية داخل البيئة (إن أمكن) — لن يمنع التشغيل إذا فشل التثبيت"
pip install --no-cache-dir aiohttp discord.py || true

# If AUTO_DOWNLOAD is true, try to install transformers & torch in the venv to enable local generation.
if [ "${AUTO_DOWNLOAD}" = "true" ] || [ "${AUTO_DOWNLOAD}" = "1" ]; then
  echo "AUTO_DOWNLOAD مفعل: محاولة تثبيت مكتبات transformers و torch (قد تستغرق وقتاً ومساحة)..."
  # Install transformers and friends; allow failure so bot still starts
  pip install --no-cache-dir transformers accelerate safetensors huggingface-hub || true
  # Install CPU-only torch wheel (non-fatal). Use PyTorch CPU index to avoid GPU wheels.
  pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || true
fi

echo "تشغيل البوت..."
# Use exec so signals are forwarded properly to the Python process
if [ -f "${VENV_DIR}/bin/python" ]; then
  exec "${VENV_DIR}/bin/python" bot.py
else
  exec python bot.py
fi
