"""
FAHAD - Auto-download-capable bot

Features added in this commit:
- AUTO_DOWNLOAD mode: when AUTO_DOWNLOAD=true in env, the bot will attempt to install required packages (transformers and torch)
  at runtime and download a specified small model (MODEL_NAME) from Hugging Face. This requires internet access and may take
  time/resources during the first run. No manual downloads required from you.
- The bot still keeps the lightweight Arabic fallback generator so it always responds even if the auto-download fails.
- The bot attempts the auto-install/load lazily on the first message that requires model generation.

Notes / warnings:
- Installing torch/transformers at runtime can be slow and may fail on limited hosts (Render free tier). If it fails,
  the bot falls back to the template-based generator and will continue responding.
- For more reliable and faster downloads, set an HF_TOKEN in the environment (optional).
"""

import os
import re
import asyncio
import json
import time
import subprocess
import sys
from aiohttp import web
import discord
import random

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

# ---------- Dynamic installer & loader ----------
async def run_subprocess(cmd, timeout=900):
    """Run subprocess asynchronously and stream output to logs."""
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
    # Try to import first
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception:
        pass

    # If AUTO_DOWNLOAD disabled, skip
    if not AUTO_DOWNLOAD:
        return False

    # Attempt to pip install transformers and torch
    print("AUTO_DOWNLOAD enabled: attempting to install transformers and torch. This may take several minutes...")
    # Prefer installing a CPU-only torch wheel where available to reduce size; let pip choose the best
    cmds = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "transformers", "accelerate", "safetensors"],
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
    ]
    for cmd in cmds:
        code, output = await run_subprocess(cmd, timeout=1200)
        print('CMD:', ' '.join(cmd), 'RETURN:', code)
        print(output[:2000])
        if code != 0:
            print('One of the install steps failed; aborting automatic install.')
            return False

    # Try import again
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        TRANSFORMERS_AVAILABLE = True
        return True
    except Exception as e:
        print('Import after install failed:', e)
        return False

async def load_transformers_model():
    """Lazy-load the small transformers model."""
    global MODEL, TOKENIZER, MODEL_DEVICE, MODEL_LOADING
    if MODEL is not None:
        return True
    if MODEL_LOADING:
        # another coro is loading
        return False
    MODEL_LOADING = True
    ok = await ensure_transformers_installed()
    if not ok:
        MODEL_LOADING = False
        return False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"Downloading model {MODEL_NAME} from Hugging Face (if not cached)...")
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            MODEL_DEVICE = 'cuda'
            MODEL.to('cuda')
        else:
            MODEL_DEVICE = 'cpu'
            MODEL.to('cpu')
        MODEL_LOADING = False
        print('Model loaded successfully on', MODEL_DEVICE)
        return True
    except Exception as e:
        print('Failed to load transformers model:', e)
        MODEL = None
        TOKENIZER = None
        MODEL_LOADING = False
        return False

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

# ---------- Simple transformers generation (if model loaded) ----------

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
        return None

# ---------- Discord events ----------

WHO_MADE_PATTERNS = [r"\bمن\s+صنعك\b", r"\bمن\s+سواك\b", r"\bمين\s+سواك\b", r"who\s+made\s+you\b"]

@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print(f"AUTO_DOWNLOAD={AUTO_DOWNLOAD}, MODEL_NAME={MODEL_NAME}, REPLY_ALL={REPLY_ALL}")

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
            # reply-to-bot
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

    # Attempt to use transformers model if AUTO_DOWNLOAD enabled
    reply = None
    if AUTO_DOWNLOAD:
        # try to load model lazily (non-blocking load attempt)
        if MODEL is None and not MODEL_LOADING:
            # schedule model loading but don't block overly long in case of failure
            # We load synchronously here to ensure we can use it for the current message if possible
            loaded = await load_transformers_model()
            if loaded:
                # generate
                prompt = f"المستخدم: {message.content}\nالمساعد:"
                gen = generate_with_transformers(prompt, max_new_tokens=128, temperature=0.7)
                if gen:
                    reply = gen
        elif MODEL is not None:
            prompt = f"المستخدم: {message.content}\nالمساعد:"
            gen = generate_with_transformers(prompt, max_new_tokens=128, temperature=0.7)
            if gen:
                reply = gen

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
