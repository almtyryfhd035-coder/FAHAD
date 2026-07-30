# FAHAD - Discord AI Bot (tuned for programming help with free open-source models)
# Updated bot.py: uses Hugging Face Inference (recommended model: bigcode/starcoder)
# - Detects code-related requests and switches prompt to programming-expert mode
# - Keeps conversation history per channel
# - Works with only DISCORD_BOT_TOKEN using local fallback, but for strong coding AI set HF_API_TOKEN

import os
import re
import asyncio
import json
from aiohttp import web, ClientSession
import discord
import random

# Read config from environment
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# Recommended free/open-source model for programming: bigcode/starcoder
MODEL_NAME = os.getenv("MODEL_NAME", "bigcode/starcoder")
PORT = int(os.getenv("PORT", "8080"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))  # number of previous messages to include
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.2"))  # lower for deterministic code outputs
HF_MAX_TOKENS = int(os.getenv("HF_MAX_TOKENS", "1024"))

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# In-memory conversation history: channel_id -> list of (role, text)
conversation_history = {}

# System instructions
SYSTEM_CHAT = (
    "أنت مساعد ذكي تتحدث باللهجة السعودية العامية. ردودك قصيرة ومباشرة وتعطي أمثلة بسيطة عند الحاجة. "
    "لا تطلب معلومات سرية من المستخدم."
)

SYSTEM_CODE = (
    "أنت خبير برمجي محترف وتساعد بشرح، تصحيح، وكتابة الشيفرات. "
    "إذا طلب المستخدم كود أعطه مثالاً عملياً يعمل. اذكر اللغة، وضع ملاحظات قصيرة باللهجة السعودية. "
    "تفضّل أن تكون دقيقاً ومباشراً وتهتم بالأداء والأمان."
)

EXAMPLE_CHAT = (
    "مثال:\nالمستخدم: وش أفضل لغة للتعلم في 2026؟\nالمساعد: بالنسبة للمبتدئين، بايثون ممتازة.."
)

EXAMPLE_CODE = (
    "مثال:\nالمستخدم: اكتب لي دالة Python تقلب مصفوفة.\n"
    "المساعد: ```python\ndef reverse_list(a):\n    return a[::-1]\n```\n" 
    "شرح: هذي دالة بسيطة تستخدم slicing."
)

FALLBACK_TEMPLATES = [
    "يا هلا، {ans}",
    "أبد، {ans}",
    "أكيد، {ans}",
]


def is_code_request(text: str) -> bool:
    """Very simple heuristics to detect code/programming requests."""
    txt = text.lower()
    # Arabic cues
    if any(k in txt for k in ["كود", "برمج", "بايثون", "python", "جافا", "java", "javascript", "js", "function", "دالة", "سطر"]):
        return True
    # code block or inline code markers
    if '```' in text or '`' in text:
        return True
    # presence of common code chars and patterns
    if re.search(r"\bdef\s+\w+\(|\bfunction\s+\w+\(|\bconsole\.log\(|\{|\}|;<", text):
        return True
    return False


def local_fallback_response(user_text: str) -> str:
    txt = user_text.lower()
    if re.search(r"\b(هلا|مرحبا|مراحب)\b", txt):
        return random.choice(["يا هلا والله، وش تبي؟", "هلا، كيف أقدر أساعدك؟"])
    if re.search(r"\b(شكرا|مشكور|تسلم)\b", txt):
        return random.choice(["العفو، بأي خدمة.", "لا شكر على واجب."])
    if is_code_request(user_text):
        return "أقدر أساعدك بالبرمجة، علمني اللغة أو المشكلة اللي تواجهك وبكتب لك مثال عملي مختصر." 
    generic = random.choice([
        "أقدر أساعدك بهالموضوع، عطِني تفاصيل أكثر.",
        "بشرحلك خطوة بخطوة لو تبغى.",
    ])
    return random.choice(FALLBACK_TEMPLATES).format(ans=generic)


def build_prompt(channel_id: int, user_text: str, code_mode: bool) -> str:
    """Construct a prompt including system instructions, short examples, and recent history."""
    history = conversation_history.get(channel_id, [])[-MAX_HISTORY:]
    parts = []
    if code_mode:
        parts.append(SYSTEM_CODE)
        parts.append(EXAMPLE_CODE)
    else:
        parts.append(SYSTEM_CHAT)
        parts.append(EXAMPLE_CHAT)

    if history:
        hist_lines = []
        for role, text in history:
            if role == 'user':
                hist_lines.append(f"المستخدم: {text}")
            else:
                hist_lines.append(f"المساعد: {text}")
        parts.append("\n".join(hist_lines))

    parts.append(f"المستخدم: {user_text}\nالمساعد:")
    return "\n\n".join(parts)

async def call_hf_api(prompt: str) -> str:
    """Call Hugging Face Inference API with parameters optimized for code assistance."""
    if not HF_API_TOKEN:
        return None

    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Accept": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"temperature": HF_TEMPERATURE, "max_new_tokens": HF_MAX_TOKENS},
        "options": {"wait_for_model": True}
    }

    try:
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=180) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text)
                except Exception:
                    return f"خطأ في استجابة الـ API: {text}"

                if isinstance(data, dict) and data.get("error"):
                    return f"خطأ من Hugging Face: {data.get('error')}"
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict) and 'generated_text' in first:
                        return first['generated_text'].strip()
                    if isinstance(first, str):
                        return first.strip()
                if isinstance(data, dict) and 'generated_text' in data:
                    return data['generated_text'].strip()
                return str(data)
    except asyncio.TimeoutError:
        return "انتهت مهلة الاتصال بمزود الخدمة. جرّب لاحقاً."
    except Exception as e:
        return f"حصل خطأ أثناء الاتصال بمزود الخدمة: {e}"


def strip_mention(content: str, bot_user: discord.User) -> str:
    content = re.sub(rf"<@!?{bot_user.id}>", "", content)
    content = content.replace(bot_user.name, "")
    return content.strip()

@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id: {client.user.id})")
    print("------")

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if client.user in message.mentions:
        user_text = strip_mention(message.content, client.user)
        if not user_text:
            await message.channel.send("ها؟ كيف أقدر أخدمك؟ اكتب سؤالك بعد المنشن.")
            return

        ch_id = message.channel.id
        conversation_history.setdefault(ch_id, []).append(('user', user_text))

        code_mode = is_code_request(user_text)
        prompt = build_prompt(ch_id, user_text, code_mode)

        async with message.channel.typing():
            reply = None
            if HF_API_TOKEN:
                reply = await call_hf_api(prompt)

            if not reply:
                reply = local_fallback_response(user_text)

            conversation_history.setdefault(ch_id, []).append(('assistant', reply))

            if len(reply) > 1900:
                reply = reply[:1900] + "..."
            try:
                await message.reply(reply)
            except Exception:
                await message.channel.send(reply)

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
