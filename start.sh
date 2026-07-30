#!/usr/bin/env bash
# start.sh - start script for FAHAD bot (venv-enabled)
# This version creates a temporary virtualenv to run pip as non-root and avoid pip warnings.
# It installs minimal deps into the venv (non-fatal if fails), then execs the bot with the venv Python.

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

echo "تشغيل البوت..."
# Use exec so signals are forwarded properly to the Python process
if [ -f "${VENV_DIR}/bin/python" ]; then
  exec "${VENV_DIR}/bin/python" bot.py
else
  exec python bot.py
fi
