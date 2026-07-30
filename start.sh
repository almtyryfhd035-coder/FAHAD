#!/usr/bin/env bash
set -e

# start.sh - start script for FAHAD bot
# Behavior:
# 1) Install minimal dependencies quickly (non-fatal if fails)
# 2) Start the bot (bot.py). The bot code performs background model download if AUTO_DOWNLOAD=true.
# Use this as Railway Start Command: bash start.sh

echo "بدء تثبيت الحزم الأساسية..."
# Install minimal packages quickly; allow failure so the bot still starts and uses fallback
pip install --no-cache-dir aiohttp discord.py || true

echo "تشغيل البوت..."
# exec so signals are forwarded to python process
exec python bot.py
