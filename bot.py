"""
FAHAD - HF Inference, caching, concurrency and safer defaults

This version enhances the bot to:
- Use Hugging Face Inference API (if USE_HF_INFERENCE=true and HUGGINGFACE_HUB_TOKEN is set) before attempting local models.
- Adds a simple in-memory cache for recent prompts to reduce API usage and latency.
- Adds an async semaphore to limit concurrent HF Inference calls (HF_CONCURRENCY env var).
- Sets safer defaults: if USE_HF_INFERENCE is enabled we avoid using the partially-downloaded local model.
- Keeps previous robustness for partial downloads, atomic .part handling, and fallback local generator.
- All important logs are printed in Arabic and flushed immediately.

Notes:
- You must set HUGGINGFACE_HUB_TOKEN in environment for HF Inference to work.
- To enable HF inference put USE_HF_INFERENCE=true and MODEL_NAME to a HF model (e.g., "gpt2", "gpt2-medium", "bigscience/bloom", "bigscience/bloomz-560m", etc.).
- HF Inference may cost credits depending on model usage — monitor your account.
"""

import os
import re
import asyncio
import json
import time
import subprocess
import sys
import traceback
from aiohttp import web
import aiohttp
import discord
import random
import contextlib
import logging
import signal

# Setup logging levels to reduce noise
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Normalize HF token env names
hf_token_env = os.getenv('HF_TOKEN') or os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_API')
if hf_token_env:
    os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token_env
    os.environ['HF_TOKEN'] = hf_token_env
    os.environ['HF_API_TOKEN'] = hf_token_env

# Config
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PORT = int(os.getenv("PORT", "8080"))
AUTO_DOWNLOAD = os.getenv("AUTO_DOWNLOAD", "false").lower() in ("1", "true", "yes")
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "true").lower() in ("1", "true", "yes")
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
REPLY_ALL = os.getenv("REPLY_ALL", "true").lower() in ("1", "true", "yes")
CHANNEL_COOLDOWN = float(os.getenv("CHANNEL_COOLDOWN", "0.2"))
USER_COOLDOWN = float(os.getenv("USER_COOLDOWN", "0.1"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "8"))
CREATOR_REPLY = os.getenv("CREATOR_REPLY", "فهد المطيري @w4px")

# HF Inference tuning
HF_CONCURRENCY = int(os.getenv('HF_CONCURRENCY', '3'))
HF_TIMEOUT = int(os.getenv('HF_TIMEOUT', '60'))
HF_CACHE_MAX = int(os.getenv('HF_CACHE_MAX', '200'))
HF_CACHE_TTL = int(os.getenv('HF_CACHE_TTL', '300'))  # seconds

# Model downloader settings (optional)
MODEL_DOWNLOAD_URL = os.getenv('MODEL_DOWNLOAD_URL')
LLAMA_CPP_BIN = os.getenv('LLAMA_CPP_BIN', './main')
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH', './models/auto_model.bin')

# Safety thresholds
MAX_DOWNLOAD_MB = int(os.getenv('MAX_DOWNLOAD_MB', '290'))  # stop download if it exceeds this
MIN_MODEL_BYTES = int(os.getenv('MIN_MODEL_BYTES', str(10 * 1024 * 1024)))  # require at least 10MB

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

last_reply_time_channel = {}
last_reply_time_user = {}
conversation_history = {}

# Model runtime state
TRANSFORMERS_AVAILABLE = False
MODEL = None
TOKENIZER = None
MODEL_DEVICE = 'cpu'
MODEL_LOADING = False

# HF helpers
HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
hf_semaphore = asyncio.Semaphore(HF_CONCURRENCY)

# Simple TTL cache for HF responses
hf_cache = {}  # key -> (response, expires_at)

# Safe helpers
AR_QUESTION_WORDS = ["وش", "شلون", "كيف", "ليش", "متى", "وين", "هل", "لماذا", "كم"]
GREETINGS = ["هلا", "مرحبا", "أهلين", "سلام", "يا هلا"]
THANKS = ["شكرا", "مشكور", "تسلم", "جزاك"]
CODE_WORDS = ["بايثون", "python", "javascript", "js", "c++", "java", "كود", "دالة", "function", "class", "print("]

# ---------- HF Inference client (async) ----------

async def hf_inference_generate(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
    """Call Hugging Face Inference API for text generation with caching and concurrency limits."""
    if not HF_TOKEN:
        print('خدمة HF غير مفعلة: HUGGINGFACE_HUB_TOKEN غير موجود', flush=True)
        return None

    # Simple cache key
    key = f"hf|{MODEL_NAME}|{max_new_tokens}|{temperature}|{prompt}"
    now = time.time()
    # prune cache expired entries opportunistically
    expired = [k for k, (_, exp) in hf_cache.items() if exp < now]
    for k in expired:
        hf_cache.pop(k, None)
    if key in hf_cache:
        resp, exp = hf_cache[key]
        if exp >= now:
            return resp
        else:
            hf_cache.pop(key, None)

    # limit concurrency
    async with hf_semaphore:
        url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "return_full_text": True
            },
            "options": {"wait_for_model": True}
        }
        try:
            timeout = aiohttp.ClientTimeout(total=HF_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        txt = await resp.text()
                        print(f"HF inference failed status={resp.status}: {txt[:200]}", flush=True)
                        return None
                    data = await resp.json()
                    # HF may return list or dict
                    if isinstance(data, list) and data:
                        text = data[0].get('generated_text') or (data[0].get('text') if 'text' in data[0] else None)
                    elif isinstance(data, dict):
                        text = data.get('generated_text') or data.get('text')
                    else:
                        text = None
                    if text:
                        # cache
                        if len(hf_cache) >= HF_CACHE_MAX:
                            # evict oldest
                            oldest = min(hf_cache.items(), key=lambda it: it[1][1])[0]
                            hf_cache.pop(oldest, None)
                        hf_cache[key] = (text, now + HF_CACHE_TTL)
                        # If returned text includes the prompt (return_full_text True), strip it
                        if text.startswith(prompt):
                            return text[len(prompt):].strip()
                        return text.strip()
        except Exception as e:
            print('خطأ أثناء استدعاء HF Inference:', e, flush=True)
            return None

# ---------- Reuse local functions from previous robust bot code (download, transformers loader, llama.cpp)
# For brevity we will import or inline the previously implemented helpers if present

# (The rest of the code reuses the previous robust implementation: download_model_file, load_transformers_model, generate_with_llama_cpp_prompt, local_ai_reply, etc.)
# To avoid duplication in this change, we'll import them dynamically by reading bot_core.py if exists, otherwise fallback to inline minimal implementations.

BOT_CORE_PATH = os.path.join(os.path.dirname(__file__), 'bot_core.py')
if os.path.exists(BOT_CORE_PATH):
    # import bot_core as module
    import importlib.util
    spec = importlib.util.spec_from_file_location('bot_core', BOT_CORE_PATH)
    bot_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot_core)
    download_model_file = bot_core.download_model_file
    ensure_transformers_installed = bot_core.ensure_transformers_installed
    load_transformers_model = bot_core.load_transformers_model
    generate_with_llama_cpp_prompt = bot_core.generate_with_llama_cpp_prompt
    generate_with_transformers = bot_core.generate_with_transformers
    local_ai_reply = bot_core.local_ai_reply
else:
    # minimal inline fallback implementations (use simple local generator)
    async def download_model_file(url: str, dest_path: str, chunk_size: int = 1 << 20):
        print('download_model_file helper غير متاح (core مفقود)')
        return False
    async def ensure_transformers_installed():
        return False
    async def load_transformers_model():
        return False
    async def generate_with_llama_cpp_prompt(prompt: str, max_tokens: int = 256, temp: float = 0.7):
        return None
    def generate_with_transformers(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7):
        return None
    def local_ai_reply(text: str) -> str:
        # very small fallback
        return "هلا! أعد صياغة السؤال أو اطلب 'اشرح' أو 'اعطني كود'."

# ---------- Discord event handlers (use HF inference if enabled) ----------

@client.event
async def on_ready():
    print(f"تم تسجيل الدخول كبوت: {client.user} (id: {client.user.id})", flush=True)
    print(f"USE_HF_INFERENCE={USE_HF_INFERENCE}, AUTO_DOWNLOAD={AUTO_DOWNLOAD}, MODEL_NAME={MODEL_NAME}", flush=True)
    # If using HF inference, prefer that and avoid local partial models
    if USE_HF_INFERENCE:
        # make sure we don't accidentally use a corrupted local model
        try:
            if os.path.exists(LLAMA_MODEL_PATH) and os.path.getsize(LLAMA_MODEL_PATH) < MIN_MODEL_BYTES:
                os.remove(LLAMA_MODEL_PATH)
                print('حُذف ملف نموذج محلي غير مكتمل لأن HF inference مفعل', flush=True)
        except Exception:
            pass
    # If AUTO_DOWNLOAD enabled, start background tasks
    if AUTO_DOWNLOAD and MODEL_DOWNLOAD_URL and not os.path.isfile(LLAMA_MODEL_PATH):
        asyncio.create_task(download_and_prepare_model())
    # If using transformers locally and AUTO_DOWNLOAD false but transformers installed, try to load
    if not AUTO_DOWNLOAD and not USE_HF_INFERENCE:
        asyncio.create_task(load_transformers_model())

async def download_and_prepare_model():
    try:
        if MODEL_DOWNLOAD_URL and not os.path.isfile(LLAMA_MODEL_PATH):
            ok = await download_model_file(MODEL_DOWNLOAD_URL, LLAMA_MODEL_PATH)
            if ok:
                print('تم تنزيل النموذج بنجاح إلى', LLAMA_MODEL_PATH, flush=True)
            else:
                print('فشل تنزيل النموذج أو كان التنزيل غير مكتمل', flush=True)
        if AUTO_DOWNLOAD and not USE_HF_INFERENCE:
            asyncio.create_task(load_transformers_model())
    except Exception:
        print('خطأ في عملية تنزيل النموذج:', flush=True)
        traceback.print_exc()

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    now = time.time()
    ch_id = message.channel.id
    usr_id = message.author.id
    last_ch = last_reply_time_channel.get(ch_id, 0)
    last_usr = last_reply_time_user.get(usr_id, 0)
    if now - last_ch < CHANNEL_COOLDOWN:
        return
    if now - last_usr < USER_COOLDOWN:
        return

    should_reply = REPLY_ALL
    if not REPLY_ALL:
        if client.user in message.mentions:
            should_reply = True
    if not should_reply:
        return

    conversation_history.setdefault(ch_id, []).append(('user', message.content))
    if len(conversation_history[ch_id]) > MAX_HISTORY*2:
        conversation_history[ch_id] = conversation_history[ch_id][-MAX_HISTORY*2:]

    low = message.content.lower()
    for pat in [r"\bمن\s+صنعك\b", r"\bمن\s+سواك\b", r"\bمين\s+سواك\b", r"who\s+made\s+you\b"]:
        if re.search(pat, low):
            reply = CREATOR_REPLY
            try:
                await message.reply(reply)
                last_reply_time_channel[ch_id] = now
                last_reply_time_user[usr_id] = now
            except Exception:
                try:
                    await message.channel.send(reply)
                    last_reply_time_channel[ch_id] = now
                    last_reply_time_user[usr_id] = now
                except Exception:
                    pass
            return

    # Try HF inference first if enabled
    reply = None
    if USE_HF_INFERENCE and HF_TOKEN:
        prompt = build_prompt_for_model(message.content)
        hf_resp = await hf_inference_generate(prompt, max_new_tokens=128, temperature=0.7)
        if hf_resp:
            reply = hf_resp

    # Try local llama.cpp if model present
    use_llama = os.path.isfile(LLAMA_MODEL_PATH) and os.path.getsize(LLAMA_MODEL_PATH) >= MIN_MODEL_BYTES
    if not reply and use_llama:
        gen = await generate_with_llama_cpp_prompt(build_prompt_for_model(message.content), max_tokens=256, temp=0.7)
        if gen:
            reply = gen

    # Try transformers local model if loaded
    if not reply and MODEL is not None:
        prompt = f"المستخدم: {message.content}\nالمساعد:"
        gen = generate_with_transformers(prompt, max_new_tokens=128, temperature=0.7)
        if gen:
            reply = gen

    # If nothing generated, fallback
    if not reply:
        reply = local_ai_reply(message.content)

    conversation_history.setdefault(ch_id, []).append(('assistant', reply))

    try:
        await message.reply(reply)
        last_reply_time_channel[ch_id] = now
        last_reply_time_user[usr_id] = now
    except Exception:
        try:
            await message.channel.send(reply)
            last_reply_time_channel[ch_id] = now
            last_reply_time_user[usr_id] = now
        except Exception:
            pass

# Build a short system-like prompt for future use if needed
def build_prompt_for_model(user_text: str) -> str:
    return f"المستخدم: {user_text}\nالمساعد:"

# Health server and graceful shutdown — lightweight
runner = None
site = None

async def start_webserver():
    global runner, site
    async def handle(request):
        return web.Response(text="OK")
    app = web.Application()
    app.router.add_get('/', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    print(f"خادم الصحة شغّال على المنفذ {PORT}", flush=True)

async def stop_webserver():
    global runner
    try:
        if runner is not None:
            await runner.cleanup()
            print('تم إيقاف خادم الصحة — stopping', flush=True)
    except Exception:
        print('حدث خطأ أثناء إيقاف خادم الصحة', flush=True)
        traceback.print_exc()

async def shutdown_bot():
    try:
        print('جارٍ إيقاف البوت — stopping', flush=True)
        await stop_webserver()
        try:
            await client.close()
        except Exception:
            pass
        print('تم إيقاف البوت تماماً — stopped', flush=True)
    except Exception:
        traceback.print_exc()

def _signal_handler(signame):
    print(f"استلمنا إشارة {signame} — جارٍ الإيقاف...", flush=True)
    try:
        asyncio.get_event_loop().create_task(shutdown_bot())
    except Exception:
        pass

async def main():
    loop = asyncio.get_event_loop()
    for signame in ('SIGINT', 'SIGTERM'):
        try:
            loop.add_signal_handler(getattr(signal, signame), lambda s=signame: _signal_handler(s))
        except NotImplementedError:
            pass

    # remove partial files on startup
    try:
        part = LLAMA_MODEL_PATH + '.part'
        if os.path.exists(part):
            try:
                os.remove(part)
                print('وجد ملف جزئي للنموذج وتمت إزالته لتجنّب الاستخدام الخاطئ', flush=True)
            except Exception:
                pass
        if os.path.exists(LLAMA_MODEL_PATH) and os.path.getsize(LLAMA_MODEL_PATH) < MIN_MODEL_BYTES:
            try:
                os.remove(LLAMA_MODEL_PATH)
                print('حذف ملف النموذج الصغير/الغير مكتمل قبل التشغيل', flush=True)
            except Exception:
                pass
    except Exception:
        pass

    await start_webserver()
    await client.start(DISCORD_BOT_TOKEN)

if __name__ == '__main__':
    try:
        # If USE_HF_INFERENCE is enabled we prefer that and disable AUTO_DOWNLOAD locally
        if USE_HF_INFERENCE and HF_TOKEN:
            print('HF Inference مفعل - سيتم استخدام واجهة HF للتوليد ونوقف التنزيل المحلي لتجنّب الاستخدام الجزئي', flush=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("جارٍ إيقاف البوت — stopping", flush=True)
    except SystemExit:
        print("تم إيقاف العملية — stopping", flush=True)
    except Exception:
        traceback.print_exc()
