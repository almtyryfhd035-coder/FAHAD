# FAHAD - Reply-all Discord AI-like Bot (local lightweight responder)
# This version replies to every user message (configurable) using a robust local fallback
# behavior so it works even without any external API. It still supports Hugging Face if
# HF_API_TOKEN is provided, but will not depend on it.

import os
import re
import asyncio
import json
import time
from aiohttp import web, ClientSession
import discord
import random

# Configuration from environment
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "bigcode/starcoder")
PORT = int(os.getenv("PORT", "8080"))
REPLY_ALL = os.getenv("REPLY_ALL", "true").lower() in ("1", "true", "yes")
CHANNEL_COOLDOWN = float(os.getenv("CHANNEL_COOLDOWN", "1.5"))  # seconds between bot replies per channel
USER_COOLDOWN = float(os.getenv("USER_COOLDOWN", "0.8"))      # prevent echo spam per user

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# In-memory state
last_reply_time_channel = {}   # channel_id -> timestamp
last_reply_time_user = {}      # user_id -> timestamp
conversation_history = {}      # channel_id -> [(role, text)] limited size
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))

# Simple detectors
CODE_KEYWORDS = ["def ", "class ", "import ", "console.log", "function(", "printf(", "#include", "<html", "SELECT "]
QUESTION_WORDS_AR = ["وش", "شلون", "كيف", "ليش", "متى", "وين", "هل", "لماذا", "كم"]
GREETING_WORDS = ["هلا", "مرحبا", "أهلين", "يا هلا", "سلام", "السلام"]
THANK_WORDS = ["شكرا", "مشكور", "تسلم", "جزاك"]

# Utilities
def is_code_request(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in CODE_KEYWORDS):
        return True
    if '```' in text or '`' in text:
        return True
    return False

def contains_question(text: str) -> bool:
    if '?' in text:
        return True
    t = text.lower()
    return any(w in t for w in QUESTION_WORDS_AR)

# Lightweight "smart" local responder
def local_ai_reply(user_text: str) -> str:
    text = user_text.strip()
    low = text.lower()

    # Greetings
    if any(g in low for g in GREETING_WORDS):
        return random.choice(["يا هلا والله، كيف أقدر أخدمك؟", "هلا، وش تبي؟"])

    # Thanks
    if any(tk in low for tk in THANK_WORDS):
        return random.choice(["العفو، في خدمتك!", "لا تشيل هم، أي طلب ثاني؟"])

    # Code-related
    if is_code_request(text):
        # Very small heuristics to guess language
        if 'python' in low or 'بايثون' in low or 'def ' in low:
            example = "```python\ndef reverse_list(a):\n    return a[::-1]\n```\nهذا مثال بسيط يقلّب القائمة باستخدام slicing، تبي شرح أكثر؟"
            return example
        if 'javascript' in low or 'js' in low or 'console.log' in low:
            example = "```javascript\nfunction reverseArray(a){\n  return a.slice().reverse();\n}\n```\nهذا مثال بسيط بالجافاسكربت، تحتاجه مفصّل؟"
        # generic code answer
        return "أقدر أساعدك بالبرمجة — عطني اللغة أو شرح للمشكلة وأعطيك مثال عملي مختصر."

    # Direct question
    if contains_question(text):
        # Try to extract keywords and give a short helpful reply
        # extract numbers or keywords
        keywords = re.findall(r"[\w\u0621-\u064A]+", text)
        short = " ".join(keywords[:6])
        return f"طيب فهمت السؤال عن: {short}. علمني تفاصيل شوي وبدّي أجاوبك بخطوات سهلة."

    # Math expression: safe eval (very limited)
    if re.match(r"^[0-9\s\+\-\*\/\.%()]+$", text):
        try:
            val = eval(text, {"__builtins__":None}, {})
            return f"الناتج: {val}"
        except Exception:
            return "حصل خطأ في حساب التعبير — تأكد من الصيغة." 

    # Short echo + offer
    if len(text.split()) <= 6:
        return random.choice([
            f"ممتاز، قلت: '{text}'. تبي أشرح أو أعطي مثال؟",
            f"تمام، بالنسبة لـ {text} أقدر أبدأ بشرح بسيط أو أمثلة، وش تفضّل؟",
        ])

    # Fallback: summarize / clarify
    # naive summary: take first 20 words
    words = text.split()
    summary = " ".join(words[:20]) + ("..." if len(words) > 20 else "")
    return f"أقدر أساعد بهالموضوع. ملخّص سريع: {summary}، عطني تفاصيل أو أسأل سؤال محدد وأرد عليك." 

# Optional: call HF API if available (kept non-blocking and protected)
async def call_hf_api(prompt: str) -> str:
    if not HF_API_TOKEN:
        return None
    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Accept": "application/json"}
    payload = {"inputs": prompt, "parameters": {"temperature": 0.2, "max_new_tokens": 256}, "options": {"wait_for_model": True}}
    try:
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=20) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text)
                except Exception:
                    return None
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict) and 'generated_text' in first:
                        return first['generated_text'].strip()
                    if isinstance(first, str):
                        return first.strip()
                if isinstance(data, dict) and 'generated_text' in data:
                    return data['generated_text'].strip()
                return None
    except Exception:
        return None

# Build a simple prompt for HF when used
def build_prompt_for_hf(user_text: str) -> str:
    return f"أجب باللغة العربية السعودية العامية وباختصار، وكن مفيداً للمستخدم.\nالمستخدم: {user_text}\nالمساعد:"

@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print("FAHAD: reply-all mode is", REPLY_ALL)

@client.event
async def on_message(message: discord.Message):
    # ignore bot messages
    if message.author.bot:
        return

    # Control cooldowns to avoid spam
    now = time.time()
    ch_id = message.channel.id
    usr_id = message.author.id
    last_ch = last_reply_time_channel.get(ch_id, 0)
    last_usr = last_reply_time_user.get(usr_id, 0)
    if now - last_ch < CHANNEL_COOLDOWN:
        # skip replying on channel cooldown
        return
    if now - last_usr < USER_COOLDOWN:
        return

    # If REPLY_ALL is False, only reply on mention (legacy behavior)
    if not REPLY_ALL and client.user not in message.mentions:
        return

    # Append to history (bounded)
    conversation_history.setdefault(ch_id, []).append(('user', message.content))
    if len(conversation_history[ch_id]) > MAX_HISTORY*2:
        conversation_history[ch_id] = conversation_history[ch_id][-MAX_HISTORY*2:]

    # Try HF first (if available), otherwise local
    reply = None
    if HF_API_TOKEN:
        prompt = build_prompt_for_hf(message.content)
        try:
            reply = await call_hf_api(prompt)
        except Exception:
            reply = None

    if not reply:
        reply = local_ai_reply(message.content)

    # save assistant reply
    conversation_history.setdefault(ch_id, []).append(('assistant', reply))

    # send reply
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
            # Give up silently to avoid crashing on send errors
            pass

# Simple health webserver for Render
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
