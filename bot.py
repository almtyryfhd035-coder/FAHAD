# FAHAD - Use local HF model if available (no external account required) with fallback
# Behavior:
# - Uses transformers + torch to download a public model from Hugging Face at startup (no HF API token required)
# - If torch/transformers aren't available or loading fails (e.g., resource limits on Render), falls back to robust local responder
# - Replies only when mentioned or when the user replies to a bot message (per your request)

import os
import re
import asyncio
import json
import time
from aiohttp import web, ClientSession
import discord
import random

# Try to import transformers and torch; if not available, will use fallback responder
USE_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_TRANSFORMERS = True
except Exception as e:
    # transformers or torch not installed or failed to import
    print("Transformers/torch not available or failed to import:", e)
    USE_TRANSFORMERS = False

# Configuration from environment
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")  # default small model (no auth required)
PORT = int(os.getenv("PORT", "8080"))
REPLY_ALL = os.getenv("REPLY_ALL", "false").lower() in ("1", "true", "yes")
REPLY_ON_MENTION_OR_REPLY = True

CHANNEL_COOLDOWN = float(os.getenv("CHANNEL_COOLDOWN", "0.8"))
USER_COOLDOWN = float(os.getenv("USER_COOLDOWN", "0.5"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "256"))

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# In-memory state
last_reply_time_channel = {}
last_reply_time_user = {}
conversation_history = {}

# Local responder (robust) - will be used if transformers unavailable or model fails
def local_ai_reply(user_text: str) -> str:
    text = user_text.strip()
    low = text.lower()
    # simple heuristics and templates for fast responses
    if re.search(r"\b(هلا|مرحبا|أهلين|سلام)\b", low) and len(text.split()) <= 5:
        return random.choice(["يا هلا! وش تبي؟", "هلا والله، كيف أقدر أساعدك؟"]) 
    if re.search(r"\b(شكرا|مشكور|تسلم)\b", low):
        return random.choice(["العفو، في خدمتك!", "لا شكر على واجب."])
    if re.search(r"\b(كود|بايثون|python|javascript|js|دالة|function|class)\b", low):
        if 'python' in low or 'بايثون' in low:
            return "```python\ndef example():\n    return 'مثال بسيط'\n```\nتبي أشرح لك؟"
        if 'javascript' in low or 'js' in low:
            return "```javascript\nfunction example(){\n  return 'مثال';\n}\n```\nتبي تعديل؟"
        return "أقدر أساعدك بالبرمجة — قل لي اللغة والمطلوب وأعطيك كود جاهز."
    if '?' in text or re.search(r"\b(وش|شلون|كيف|لماذا|متى|وين|كم)\b", low):
        words = re.findall(r"[\w\u0621-\u064A]+", text)
        topic = ' '.join(words[:6])
        return f"بالنسبة لـ {topic}، أقدر ألخص: وضّح المطلوب، عط�� مثال، وبسوّي لك خطوات." 
    if len(text.split()) <= 6:
        return random.choice([f"قلت: '{text}'. وضّح أكثر؟", "تبي شرح ولا أمثلة؟"]) 
    words = text.split()
    summary = ' '.join(words[:20]) + ("..." if len(words) > 20 else "")
    return f"أقدر أساعد بهالموضوع. ملخّص: {summary}. كيف تبي أبدأ؟"

# Transformers model placeholders
MODEL = None
TOKENIZER = None
MODEL_DEVICE = 'cpu'

async def load_model_if_possible():
    global MODEL, TOKENIZER, MODEL_DEVICE
    if not USE_TRANSFORMERS:
        print("Skipping model download — transformers/torch not available")
        return False
    try:
        print(f"Attempting to download/load model '{MODEL_NAME}' — this may take time and memory.")
        # prefer CPU to avoid trying to use GPU on Render
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # move to CPU
        if torch.cuda.is_available():
            MODEL_DEVICE = 'cuda'
            MODEL.to('cuda')
        else:
            MODEL_DEVICE = 'cpu'
            MODEL.to('cpu')
        print(f"Model {MODEL_NAME} loaded on {MODEL_DEVICE}")
        return True
    except Exception as e:
        print("Failed to load model locally:", e)
        MODEL = None
        TOKENIZER = None
        return False

def generate_with_local_model(prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
    global MODEL, TOKENIZER
    if not MODEL or not TOKENIZER:
        return None
    try:
        input_ids = TOKENIZER.encode(prompt, return_tensors='pt')
        if MODEL_DEVICE == 'cuda':
            input_ids = input_ids.to('cuda')
        # generation parameters
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        outputs = MODEL.generate(input_ids, **gen_kwargs)
        text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        # return only the generated continuation after prompt
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
    except Exception as e:
        print("Local model generation failed:", e)
        return None

# Build prompt for model — steer to Arabic Saudi dialect
def build_prompt_for_model(user_text: str) -> str:
    system = "أنت مساعد ذكي جدًا وترد باللهجة السعودية العامية عندما يُطلب. كن واضحًا ومفيدًا." 
    prompt = f"{system}\nالمستخدم: {user_text}\nالمساعد:"
    return prompt

async def is_reply_to_bot(message: discord.Message) -> bool:
    ref = message.reference
    if not ref:
        return False
    if getattr(ref, 'resolved', None) and hasattr(ref.resolved, 'author'):
        return ref.resolved.author.id == client.user.id
    try:
        referenced = await message.channel.fetch_message(ref.message_id)
        return referenced.author.id == client.user.id
    except Exception:
        return False

@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print(f"REPLY_ON_MENTION_OR_REPLY={REPLY_ON_MENTION_OR_REPLY}, REPLY_ALL={REPLY_ALL}")

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

    # Should reply only on mention or reply
    should_reply = False
    if REPLY_ALL:
        should_reply = True
    else:
        if client.user in message.mentions:
            should_reply = True
        else:
            if await is_reply_to_bot(message):
                should_reply = True
    if not should_reply:
        return

    # Save history
    conversation_history.setdefault(ch_id, []).append(('user', message.content))
    if len(conversation_history[ch_id]) > MAX_HISTORY*2:
        conversation_history[ch_id] = conversation_history[ch_id][-MAX_HISTORY*2:]

    reply = None
    # Try local transformers model generation if available
    if USE_TRANSFORMERS and MODEL is None:
        # Attempt to load model once (non-blocking awaited)
        loaded = await load_model_if_possible()
        if not loaded:
            print("Model not loaded — will use local fallback")
    if USE_TRANSFORMERS and MODEL is not None:
        try:
            prompt = build_prompt_for_model(message.content)
            gen = generate_with_local_model(prompt, max_new_tokens=HF_MAX_TOKENS, temperature=HF_TEMPERATURE)
            if gen:
                reply = gen
        except Exception as e:
            print("Error generating with local model:", e)
            reply = None

    # If still no reply, fallback
    if not reply:
        reply = local_ai_reply(message.content)

    # Save assistant reply
    conversation_history.setdefault(ch_id, []).append(('assistant', reply))

    # Send reply
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
    # Do NOT block startup on model download — attempt to load lazily when first needed
    await start_webserver()
    await client.start(DISCORD_BOT_TOKEN)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down")
