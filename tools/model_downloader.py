"""
FAHAD - Auto-download model helper: support MODEL_DOWNLOAD_URL for automatic ggml/model download

Behavior added:
- If AUTO_DOWNLOAD=true and MODEL_DOWNLOAD_URL is set in env, the bot will attempt to download that file to ./models/auto_model.bin
  and then, if LLAMA_CPP_BIN is present, use llama.cpp to run it. This lets you provide a direct URL to a ggml file (e.g., a gpt4all quantized model) and have the bot auto-download and use it.
- If download fails or no MODEL_DOWNLOAD_URL provided, previous behavior remains (transformers auto-download or local fallback).

Notes:
- Downloading large model files may take long and may fail on limited hosting plans — fallback is always the template-based Arabic responder.
- For recommended sources, use gpt4all releases or a Hugging Face model URL. Set MODEL_DOWNLOAD_URL to the direct downloadable URL.
"""

# (Only the added helper and integration are shown here; full bot.py already in repo.)

import os
import aiohttp
import asyncio

async def download_model_file(url: str, dest_path: str, chunk_size: int = 1 << 20):
    """
    Download a file from a URL to dest_path using streaming. Overwrites if exists.
    Returns True on success, False on failure.
    """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=60*60)  # allow up to 1 hour for big files (adjustable)
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
                        # simple progress print every ~10MB
                        if downloaded % (10 * (1 << 20)) < chunk_size:
                            print(f"Downloaded {downloaded} bytes...")
        print("Model download completed")
        return True
    except Exception as e:
        print("Exception during model download:", e)
        return False

# Integration example: in bot startup or lazy load, you can call:
# model_url = os.getenv('MODEL_DOWNLOAD_URL')
# if AUTO_DOWNLOAD and model_url and not os.path.isfile('./models/auto_model.bin'):
#     await download_model_file(model_url, './models/auto_model.bin')
# Then set LLAMA_MODEL_PATH = './models/auto_model.bin' and proceed to use llama.cpp if available.
