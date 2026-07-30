"""
FAHAD - Suppress HF warnings, Arabic logs for downloads, and HF token support guidance

Changes in this update:
- Sets huggingface_hub logger to ERROR to mute HF progress/warnings.
- Reads HF token env vars (HF_TOKEN, HF_API_TOKEN, HUGGINGFACE_HUB_TOKEN) and injects into environment for faster authenticated downloads.
- Replaced key log messages about downloading/loading models with concise Arabic messages so Railway logs show meaningful Arabic lines (e.g., "بدء تنزيل النموذج...", "اكتمل تنزيل النموذج: <path>").
- Kept robust fallback behavior: if download or load fails the bot continues responding with the Arabic fallback.
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
import discord
import random
import contextlib
import logging

# Mute Hugging Face hub verbose logs (progress bars/warnings) and set level to ERROR
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# If user provided any HF token variants, normalize them so HF libraries use them
hf_token_env = os.getenv('HF_TOKEN') or os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_API')
if hf_token_env:
    os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token_env
    os.environ['HF_TOKEN'] = hf_token_env
    os.environ['HF_API_TOKEN'] = hf_token_env

# Config
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PORT = int(os.getenv("PORT", "8080"))
# If AUTO_DOWNLOAD=true, bot will try to pip install and download a small model automatically
AUTO_DOWNLOAD = os.getenv("AUTO_DOWNLOAD", "true").lower() in ("1", "true", "yes")
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")  # small default model
REPLY_ALL = os.getenv("REPLY_ALL", "true").lower() in ("1", "true", "yes")
CHANNEL_COOLDOWN = float(os.getenv("CHANNEL_COOLDOWN", "0.2"))
USER_COOLDOWN = float(os.getenv("USER_COOLDOWN", "0.1"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "8"))
CREATOR_REPLY = os.getenv("CREATOR_REPLY", "فهد المطيري @w4px")

# Model downloader settings (optional)
MODEL_DOWNLOAD_URL = os.getenv('MODEL_DOWNLOAD_URL')
LLAMA_CPP_BIN = os.getenv('LLAMA_CPP_BIN', './main')
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH', './models/auto_model.bin')

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

# Safe helpers
AR_QUESTION_WORDS = ["وش", "شلون", "كيف", "ليش", "متى", "وين", "هل", "لماذا", "كم"]
GREETINGS = ["هلا", "مرحبا", "أهلين", "سلام", "يا هلا"]
THANKS = ["شكرا", "مشكور", "تسلم", "جزاك"]
CODE_WORDS = ["بايثون", "python", "javascript", "js", "c++", "java", "كود", "دالة", "function", "class", "print("]

# ---------- Model download helper (Arabic logging) ----------
import aiohttp

async def download_model_file(url: str, dest_path: str, chunk_size: int = 1 << 20):
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=60*60)  # allow up to 1 hour
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"فشل تنزيل النموذج، رمز HTTP: {resp.status}")
                    return False
                total = resp.headers.get('Content-Length')
                print(f"بدء تنزيل النموذج... الحجم المتوقع: {total if total else 'غير معروف'}")
                with open(dest_path, 'wb') as f:
                    downloaded = 0
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (10 * (1 << 20)) < chunk_size:
                            print(f"تم تنزيل {downloaded // (1<<20)} ميجابايت...")
        print(f"اكتمل تنزيل النموذج وحُفظ في: {dest_path}")
        return True
    except Exception as e:
        print("خطأ أثناء تنزيل النموذج:", e)
        return False

# ---------- Dynamic installer & loader with suppressed verbosity and Arabic logs ----------
async def run_subprocess(cmd, timeout=900):
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors='ignore') if stdout else ''
        err = stderr.decode(errors='ignore') if stderr else ''
        return proc.returncode, out + ('\n' + err if err else '')
    except Exception as e:
        return 1, str(e)

async def ensure_transformers_installed():
    global TRANSFORMERS_AVAILABLE
    if TRANSFORMERS_AVAILABLE:
        return True
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception:
        pass

    if not AUTO_DOWNLOAD:
        print("تنزيل تلقائي معطّل (AUTO_DOWNLOAD=false)")
        return False

    print("AUTO_DOWNLOAD مفعّل: جاري محاولة تثبيت مكتبات transformers و torch، قد يستغرق هذا عدة دقائق...")
    cmds = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "transformers", "accelerate", "safetensors"],
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
    ]
    for cmd in cmds:
        code, output = await run_subprocess(cmd, timeout=1200)
        print('تشغيل الأمر:', ' '.join(cmd), 'النتيجة:', code)
        print(output[:2000])
        if code != 0:
            print('فشل أحد خطوات التثبيت؛ سيتم إيقاف المحاولة التلقائية.')
            return False

    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception as e:
        print('فشل الاستيراد بعد التثبيت:', e)
        return False

async def load_transformers_model():
    global MODEL, TOKENIZER, MODEL_DEVICE, MODEL_LOADING
    if MODEL is not None:
        return True
    if MODEL_LOADING:
        return False
    MODEL_LOADING = True
    ok = await ensure_transformers_installed()
    if not ok:
        MODEL_LOADING = False
        print('مكتبات transformers غير متوفرة أو فشل التثبيت')
        return False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"جاري تحميل/تنزيل النموذج {MODEL_NAME} من Hugging Face (ربما بدون طباعة تقدم)...")
        # suppress stdout/stderr (tqdm/progress bars)
        devnull = open(os.devnull, 'w')
        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
                MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        finally:
            devnull.close()
        if 'cuda' in str(getattr(MODEL, 'device', '')) or (hasattr(torch, 'cuda') and torch.cuda.is_available()):
            MODEL_DEVICE = 'cuda'
            MODEL.to('cuda')
        else:
            MODEL_DEVICE = 'cpu'
            MODEL.to('cpu')
        MODEL_LOADING = False
        print('اكتمل تحميل النموذج (transformers) بنجاح')
        return True
    except Exception as e:
        MODEL = None
        TOKENIZER = None
        MODEL_LOADING = False
        print('فشل تحميل نموذج transformers:', e)
        traceback.print_exc()
        return False

# ---------- llama.cpp integration (capture and filter progress output) with Arabic logs ----------
async def generate_with_llama_cpp_prompt(prompt: str, max_tokens: int = 256, temp: float = 0.7) -> str:
    bin_path = LLAMA_CPP_BIN
    model_path = LLAMA_MODEL_PATH
    if not os.path.isfile(bin_path):
        alt = os.path.join('llama.cpp', 'main')
        if os.path.isfile(alt):
            bin_path = alt
        else:
            print('لم يتم العثور على ملف التنفيذ llama.cpp عند المسار المحدد')
            return None
    if not os.path.isfile(model_path):
        print('ملف النموذج غير موجود في المسار:', model_path)
        return None

    cmd = [bin_path, '-m', model_path, '--threads', '4', '--temp', str(temp), '--n_predict', str(max_tokens), '--prompt', prompt]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        text = stdout.decode(errors='ignore') if stdout else ''
        # Filter out noisy progress lines like "Loading weights:" or percentage bars
        lines = []
        for ln in text.splitlines():
            if re.search(r'Loading weights:|\d+%|\[[-= >]+\]', ln):
                continue
            lines.append(ln)
        filtered_text = '\n'.join(lines).strip()
        if prompt and filtered_text.startswith(prompt):
            return filtered_text[len(prompt):].strip()
        return filtered_text if filtered_text else None
    except Exception as e:
        print('خطأ أثناء تشغيل llama.cpp:', e)
        traceback.print_exc()
        return None

# ---------- Simple local generator (fallback) ----------

def safe_eval(expr: str):
    if not re.match(r"^[0-9\s\+\-\*\/\.%()]+$", expr):
        return None
    try:
        return eval(expr, {"__builtins__":None}, {})
    except Exception:
        return None

def local_ai_reply(text: str) -> str:
    text = text.strip()
    low = text.lower()
    if re.search(r"\b(من\s*(سواك|صنعك|سوّاك)|مين\s+سواك)\b", low):
        return CREATOR_REPLY
    if any(g in low for g in GREETINGS) and len(text.split()) <= 6:
        return random.choice(["يا هلا! كيف أقدر أساعدك؟", "هلا، وش اللي تبيه؟"])
    if any(t in low for t in THANKS):
        return random.choice(["العفو، في خدمتك!", "لا شكر على واجب."])
    if re.match(r"^[0-9\s\+\-\*\/\.%()]+$", text):
        r = safe_eval(text)
        return f"الناتج: {r}" if r is not None else "ما قدرت أحسب هذا التعبير."
    if any(w in low for w in CODE_WORDS):
        if 'python' in low or 'بايثون' in low:
            return ("```python\ndef reverse_list(a):\n    return a[::-1]\n```")
        if 'javascript' in low or 'js' in low:
            return ("```javascript\nfunction reverseArray(a){\n  return a.slice().reverse();\n}\n```")
        return "قل لي اللغة والمطلوب وسأعطيك كود جاهز مع شرح."
    if '?' in text or any(q in low for q in AR_QUESTION_WORDS):
        words = re.findall(r"[\w\u0621-\u064A]+", text)
        topic = ' '.join(words[:8])
        return (f"بالنسبة لـ {topic}، خلّني أشرح بالخطوات: 1) حدد المطلوب 2) أعطني مثال 3) أقدملك حل. تبيني أبدأ؟")
    if len(text.split()) <= 6:
        return random.choice([f"قلت: '{text}'. وضّح أكثر؟", "تبي شرح ولا مثال؟"]) 
    wc = len(text.split())
    if wc > 30:
        first = ' '.join(text.split()[:20])
        last = ' '.join(text.split()[-8:])
        return f"ملخّص سريع: {first} ... {last}\nتبي تفصيل أو أمثلة؟"
    return "أقدر أساعدك — أعطني تفاصيل أكثر أو اطلب 'اشرح' أو 'اعطني كود'"

# ---------- Transformers generation helper ----------

def generate_with_transformers(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
    global MODEL, TOKENIZER, MODEL_DEVICE
    if MODEL is None or TOKENIZER is None:
        return None
    try:
        import torch
        input_ids = TOKENIZER.encode(prompt, return_tensors='pt')
        if MODEL_DEVICE == 'cuda':
            input_ids = input_ids.to('cuda')
        outputs = MODEL.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
    except Exception as e:
        print('خطأ أثناء توليد الرد من transformers:', e)
        traceback.print_exc()
        return None

# ---------- Discord events ----------

WHO_MADE_PATTERNS = [r"\bمن\s+صنعك\b", r"\bمن\s+سواك\b", r"\bمين\s+سواك\b", r"who\s+made\s+you\b"]

@client.event
async def on_ready():
    print(f"تم تسجيل الدخول كبوت: {client.user} (id: {client.user.id})")
    print(f"AUTO_DOWNLOAD={AUTO_DOWNLOAD}, MODEL_NAME={MODEL_NAME}, REPLY_ALL={REPLY_ALL}")
    # If MODEL_DOWNLOAD_URL is set, attempt to download in background
    if AUTO_DOWNLOAD and MODEL_DOWNLOAD_URL and not os.path.isfile(LLAMA_MODEL_PATH):
        print('تم جدولة تنزيل النموذج في الخلفية...')
        asyncio.create_task(download_and_prepare_model())

async def download_and_prepare_model():
    # download model if URL provided
    try:
        if MODEL_DOWNLOAD_URL and not os.path.isfile(LLAMA_MODEL_PATH):
            ok = await download_model_file(MODEL_DOWNLOAD_URL, LLAMA_MODEL_PATH)
            if ok:
                print('تم تنزيل النموذج بنجاح إلى', LLAMA_MODEL_PATH)
            else:
                print('فشل تنزيل النموذج أو كان التنزيل غير مكتمل')
        # Do not block startup: schedule transformers load but don't await here
        if AUTO_DOWNLOAD:
            asyncio.create_task(load_transformers_model())
    except Exception:
        print('خطأ في عملية تنزيل النموذج:')
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
        else:
            if message.reference:
                try:
                    ref = message.reference
                    if getattr(ref, 'resolved', None) and hasattr(ref.resolved, 'author'):
                        if ref.resolved.author.id == client.user.id:
                            should_reply = True
                    else:
                        referenced = await message.channel.fetch_message(ref.message_id)
                        if referenced.author.id == client.user.id:
                            should_reply = True
                except Exception:
                    pass
    if not should_reply:
        return

    conversation_history.setdefault(ch_id, []).append(('user', message.content))
    if len(conversation_history[ch_id]) > MAX_HISTORY*2:
        conversation_history[ch_id] = conversation_history[ch_id][-MAX_HISTORY*2:]

    low = message.content.lower()
    for pat in WHO_MADE_PATTERNS:
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

    # Attempt llama.cpp generation first if model present
    reply = None
    if os.path.isfile(LLAMA_CPP_BIN) and os.path.isfile(LLAMA_MODEL_PATH):
        gen = await generate_with_llama_cpp_prompt(build_prompt_for_model(message.content), max_tokens=256, temp=0.7)
        if gen:
            reply = gen

    # Attempt transformers generation only if model is already loaded (don't await long loads here)
    if not reply and MODEL is not None:
        prompt = f"المستخدم: {message.content}\nالمساعد:"
        gen = generate_with_transformers(prompt, max_new_tokens=128, temperature=0.7)
        if gen:
            reply = gen

    # If model isn't ready and AUTO_DOWNLOAD is enabled, ensure background load is running
    if not reply and AUTO_DOWNLOAD and MODEL is None and not MODEL_LOADING:
        # schedule background load without blocking the reply
        asyncio.create_task(load_transformers_model())

    # fallback to local generator
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

# Health server
async def start_webserver():
    async def handle(request):
        return web.Response(text="OK")
    app = web.Application()
    app.router.add_get('/', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    print(f"خادم الصحة شغّال على المنفذ {PORT}")

async def main():
    await start_webserver()
    await client.start(DISCORD_BOT_TOKEN)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("جارٍ إيقاف البوت")
    except Exception:
        traceback.print_exc()
