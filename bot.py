"""
FAHAD - Auto-download-capable bot (updated)

Changes in this commit:
- Suppress/strip progress-bar and "Loading weights" noise from llama.cpp subprocess output.
- Silence transformers/from_pretrained verbose stdout/stderr using contextlib.redirect_stdout/stderr to avoid tqdm bars in logs.
- Make model loading non-blocking during on_message: schedule background load (async task) so the bot falls back to local generator immediately instead of hanging.
- Add clearer log messages for model load success/failure so you can see concise status in logs.
- Improve error handling so bot continues to reply with the fallback generator if model isn't available.
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

# ---------- Model download helper ----------
import aiohttp

async def download_model_file(url: str, dest_path: str, chunk_size: int = 1 << 20):
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=60*60)  # allow up to 1 hour
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"Model download failed, HTTP status: {resp.status}")
                    return False
                total = resp.headers.get('Content-Length')
                print(f"Starting model download: {url} -> {dest_path} (size={total})")
                with open(dest_path, 'wb') as f:
                    downloaded = 0
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (10 * (1 << 20)) < chunk_size:
                            print(f"Downloaded {downloaded} bytes...")
        print("Model download completed")
        return True
    except Exception as e:
        print("Exception during model download:", e)
        return False

# ---------- Dynamic installer & loader with suppressed verbosity ----------
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
        return False

    print("AUTO_DOWNLOAD enabled: attempting to install transformers and torch. This may take several minutes...")
    cmds = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "transformers", "accelerate", "safetensors"],
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
    ]
    for cmd in cmds:
        code, output = await run_subprocess(cmd, timeout=1200)
        print('CMD:', ' '.join(cmd), 'RETURN:', code)
        # Print truncated output to avoid huge logs
        print(output[:2000])
        if code != 0:
            print('One of the install steps failed; aborting automatic install.')
            return False

    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception as e:
        print('Import after install failed:', e)
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
        print('Transformers not available or install failed')
        return False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Downloading/loading model {MODEL_NAME} from Hugging Face (this may be quiet)...")
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
        print('Model loaded successfully')
        return True
    except Exception as e:
        MODEL = None
        TOKENIZER = None
        MODEL_LOADING = False
        print('Failed to load transformers model:', e)
        traceback.print_exc()
        return False

# ---------- llama.cpp integration (capture and filter progress output) ----------
async def generate_with_llama_cpp_prompt(prompt: str, max_tokens: int = 256, temp: float = 0.7) -> str:
    bin_path = LLAMA_CPP_BIN
    model_path = LLAMA_MODEL_PATH
    if not os.path.isfile(bin_path):
        alt = os.path.join('llama.cpp', 'main')
        if os.path.isfile(alt):
            bin_path = alt
        else:
            return None
    if not os.path.isfile(model_path):
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
        print('Error running llama.cpp subprocess:', e)
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
        print('Generation error:', e)
        traceback.print_exc()
        return None

# ---------- Discord events ----------

WHO_MADE_PATTERNS = [r"\bمن\s+صنعك\b", r"\bمن\s+سواك\b", r"\bمين\s+سواك\b", r"who\s+made\s+you\b"]

@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print(f"AUTO_DOWNLOAD={AUTO_DOWNLOAD}, MODEL_NAME={MODEL_NAME}, REPLY_ALL={REPLY_ALL}")
    # If MODEL_DOWNLOAD_URL is set, attempt to download in background
    if AUTO_DOWNLOAD and MODEL_DOWNLOAD_URL and not os.path.isfile(LLAMA_MODEL_PATH):
        print('Scheduling background model download...')
        asyncio.create_task(download_and_prepare_model())

async def download_and_prepare_model():
    # download model if URL provided
    try:
        if MODEL_DOWNLOAD_URL and not os.path.isfile(LLAMA_MODEL_PATH):
            ok = await download_model_file(MODEL_DOWNLOAD_URL, LLAMA_MODEL_PATH)
            if ok:
                print('Downloaded ggml model to', LLAMA_MODEL_PATH)
            else:
                print('Model download failed or was incomplete')
        # Do not block startup: schedule transformers load but don't await here
        if AUTO_DOWNLOAD:
            asyncio.create_task(load_transformers_model())
    except Exception:
        print('Error in download_and_prepare_model:')
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
    print(f"Health server started on port {PORT}")

async def main():
    await start_webserver()
    await client.start(DISCORD_BOT_TOKEN)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down")
    except Exception:
        traceback.print_exc()
