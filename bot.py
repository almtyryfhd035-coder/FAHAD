# Discord AI Bot with optional Hugging Face Inference API and local fallback responses
# bot.py
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
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")  # override with a chat-friendly model if available
PORT = int(os.getenv("PORT", "8080"))

if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN environment variable is required")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Prompt template to steer replies to Saudi Arabic dialect
PROMPT_SYSTEM = (
    "أنت مساعد ذكي تتحدث باللهجة السعودية العامية. \n"
    "ردودك قصيرة وودية ومباشرة، وتجاوب على سياق الرسالة بالتحديد. \n"
    "إذا سأل المستخدم عن شيء لا تعرفه، أعتذر وأعرض طريقة أخرى أو أسأل لتوضيح.\n"
    "لا تطلب من المستخدم أي مفاتيح أو معلومات حساسة.\n"
)

# Simple local fallback responder (when HF_API_TOKEN not provided)
FALLBACK_TEMPLATES = [
    "أبد، فهمت عليك. {ans}",
    "أها، طيب: {ans}",
    "أكيد، هذا اللي أقدر أقول: {ans}",
    "يا هلا، {ans}",
]

def local_fallback_response(user_text: str) -> str:
    """Generate a simple Saudi-dialect style reply based on heuristics.
    This is intentionally lightweight so you only need the Discord bot token to run.
    """
    txt = user_text.lower()
    # greetings
    if re.search(r"\b(هلا|هلابا|هلا|يا هلا|يا هلا فيك|مرحبا|مراحب)\b", txt):
        ans = random.choice(["يا هلا والله، وش تبي؟", "هلا، كيف أقدر أساعدك؟", "ياهلا، ابشر اخبرني."])
        return ans

    # thanks
    if re.search(r"\b(شكرا|مشكور|جزاك|تسلم)\b", txt):
        return random.choice(["العفو، في خدمتك!", "حياك، أي شي ثاني؟", "لا شكر على واجب."])

    # asking for help or how-to
    if re.search(r"\b(كيف|شلون|وشلون|شلون اسوي|كيف اسوي|وش اسوي)\b", txt):
        ans = "لو تشرح لي بالضبط وش تبغى أسويلك أعطيك خطوات بسيطة." 
        return random.choice(FALLBACK_TEMPLATES).format(ans=ans)

    # price/time/where
    if re.search(r"\b(كم|متى|وين|وينه|وين موقع)\b", txt):
        ans = "أعطني تفاصيل أكثر، وبرد عليك بنفس اللهجة وبسرعة." 
        return random.choice(FALLBACK_TEMPLATES).format(ans=ans)

    # default short helpful reply
    generic = random.choice([
        "أقدر أساعدك بهالموضوع، عطِني تفاصيل أكثر.",
        "معلومة حلوة، لكن بحاجة توضيح أكثر علشان أرد بدقة.",
        "أأ، فهمت بشكل عام، تبي أمثلة ولا حل خطوة بخطوة؟",
    ])
    return random.choice(FALLBACK_TEMPLATES).format(ans=generic)

async def call_hf_api(prompt: str) -> str:
    """استدعي نموذج Hugging Face Inference API وأرجع النص الناتج.
    If HF_API_TOKEN is missing, return a message to indicate fallback should be used.
    """
    if not HF_API_TOKEN:
        return None

    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Accept": "application/json"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    try:
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
                text = await resp.text()
                try:
                    data = json.loads(text)
                except Exception:
                    return f"خطأ في استجابة الـ API: {text}"

                if isinstance(data, dict) and data.get("error"):
                    return f"خطأ من Hugging Face: {data.get('error')}"
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict) and "generated_text" in first:
                        return first["generated_text"]
                    if isinstance(first, str):
                        return first
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
    # Ignore messages from bots
    if message.author.bot:
        return

    # Only respond when bot is mentioned
    if client.user in message.mentions:
        user_text = strip_mention(message.content, client.user)
        if not user_text:
            await message.channel.send("ها؟ كيف أقدر أخدمك؟ اكتب سؤالك بعد المنشن.")
            return

        prompt = PROMPT_SYSTEM + "\n" + f"المستخدم: {user_text}\nالمساعد:"
        async with message.channel.typing():
            # Try HF if token present, otherwise use local fallback
            reply = None
            if HF_API_TOKEN:
                reply = await call_hf_api(prompt)

            if not reply:
                # local fallback
                reply = local_fallback_response(user_text)

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
