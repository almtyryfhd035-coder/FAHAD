# Discord AI Bot using Hugging Face Inference API
# bot.py
import os
import re
import asyncio
import json
from aiohttp import web, ClientSession
import discord

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

async def call_hf_api(prompt: str) -> str:
    """استدعي نموذج Hugging Face Inference API وأرجع النص الناتج"""
    if not HF_API_TOKEN:
        return "خطأ: HF_API_TOKEN غير مفعّل. ضع توكن Hugging Face في متغير البيئة HF_API_TOKEN."

    url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Accept": "application/json"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    try:
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
                text = await resp.text()
                # Try to parse JSON response
                try:
                    data = json.loads(text)
                except Exception:
                    return f"خطأ في استجابة الـ API: {text}"

                # Common HF Inference return shapes: {'error': ...} or [{'generated_text': '...'}]
                if isinstance(data, dict) and data.get("error"):
                    return f"خطأ من Hugging Face: {data.get('error')}"
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict) and "generated_text" in first:
                        return first["generated_text"]
                    # Some models return text directly
                    if isinstance(first, str):
                        return first
                # fallback: string-repr
                return str(data)
    except asyncio.TimeoutError:
        return "انتهت مهلة الاتصال بمزود الخدمة. جرّب لاحقاً."
    except Exception as e:
        return f"حصل خطأ أثناء الاتصال بمزود الخدمة: {e}"


def strip_mention(content: str, bot_user: discord.User) -> str:
    # Remove mention strings like <@!id> or @BotName
    content = re.sub(rf"<@!?{bot_user.id}>", "", content)
    # Also strip bot name if present
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
        # prepare the user message (strip mention)
        user_text = strip_mention(message.content, client.user)
        if not user_text:
            await message.channel.send("ها؟ كيف أقدر أخدمك؟ اكتب سؤالك بعد المنشن.")
            return

        prompt = PROMPT_SYSTEM + "\n" + f"المستخدم: {user_text}\nالمساعد:"  
        # Indicate typing
        async with message.channel.typing():
            reply = await call_hf_api(prompt)
            # Ensure reply length under discord limit
            if len(reply) > 1900:
                reply = reply[:1900] + "..."
            try:
                await message.reply(reply)
            except Exception:
                await message.channel.send(reply)

async def start_webserver():
    # Simple health endpoint so Render considers service healthy
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
    # run webserver and discord client concurrently
    await start_webserver()
    await client.start(DISCORD_BOT_TOKEN)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down")
