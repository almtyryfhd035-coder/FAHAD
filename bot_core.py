import os
import re
import sys
import traceback
import contextlib
import aiohttp
import asyncio

# This module provides core helpers used by bot.py: download_model_file, ensure_transformers_installed,
# load_transformers_model, generate_with_llama_cpp_prompt, generate_with_transformers, local_ai_reply.

# Config defaults (bot.py will set env vars; we read them here)
MAX_DOWNLOAD_MB = int(os.getenv('MAX_DOWNLOAD_MB', '290'))
MIN_MODEL_BYTES = int(os.getenv('MIN_MODEL_BYTES', str(10 * 1024 * 1024)))
MODEL_NAME = os.getenv('MODEL_NAME', 'distilgpt2')
LLAMA_CPP_BIN = os.getenv('LLAMA_CPP_BIN', './main')
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH', './models/auto_model.bin')
AUTO_DOWNLOAD = os.getenv('AUTO_DOWNLOAD', 'false').lower() in ('1','true','yes')

# Simple local fallback reply (kept minimal to avoid duplication)
import random
CREATOR_REPLY = os.getenv('CREATOR_REPLY', 'فهد المطيري @w4px')
AR_QUESTION_WORDS = ["وش", "شلون", "كيف", "ليش", "متى", "وين", "��ل", "لماذا", "كم"]
GREETINGS = ["هلا", "مرحبا", "أهلين", "سلام", "يا هلا"]
THANKS = ["شكرا", "مشكور", "تسلم", "جزاك"]
CODE_WORDS = ["بايثون", "python", "javascript", "js", "c++", "java", "كود", "دالة", "function", "class", "print("]

async def download_model_file(url: str, dest_path: str, chunk_size: int = 1 << 20):
    """Download model to dest_path using a .part temporary file; enforce MAX_DOWNLOAD_MB and MIN_MODEL_BYTES.
    Returns True on success, False otherwise. Prints Arabic logs to stdout (bot.py expects that).
    """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=60*60)  # 1 hour
        part_path = dest_path + '.part'
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"فشل تنزيل النموذج، رمز HTTP: {resp.status}", flush=True)
                    return False
                content_length = resp.headers.get('Content-Length')
                total_expected = int(content_length) if content_length and content_length.isdigit() else None
                print(f"بدء تنزيل النموذج... الحجم المتوقع: {total_expected if total_expected else 'غير معروف'}", flush=True)
                with open(part_path, 'wb') as f:
                    downloaded = 0
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (10 * (1 << 20)) < chunk_size:
                            print(f"تم تنزيل {downloaded // (1<<20)} ميجابايت...", flush=True)
                        if downloaded > MAX_DOWNLOAD_MB * (1 << 20):
                            print(f"تجاوز تحميل النموذج الحد المسموح ({MAX_DOWNLOAD_MB} ميجابايت) — سيتم إيقاف التنزيل لحماية المساحة.", flush=True)
                            try:
                                f.close()
                            except Exception:
                                pass
                            try:
                                os.remove(part_path)
                            except Exception:
                                pass
                            return False
                final_size = os.path.getsize(part_path)
                if total_expected is not None and final_size != total_expected:
                    print(f"حجم الملف النهائي ({final_size}) لا يطابق المحتوى المتوقع ({total_expected}) — سيتم إلغاء الملف.", flush=True)
                    try:
                        os.remove(part_path)
                    except Exception:
                        pass
                    return False
                if final_size < MIN_MODEL_BYTES:
                    print(f"الملف الذي تم تنزيله صغير جداً ({final_size} بايت) — سيتم تجاهله.", flush=True)
                    try:
                        os.remove(part_path)
                    except Exception:
                        pass
                    return False
        os.replace(part_path, dest_path)
        print(f"اكتمل تنزيل النموذج وحُفظ في: {dest_path}", flush=True)
        return True
    except Exception as e:
        print("خطأ أثناء تنزيل النموذج:", e, flush=True)
        try:
            if os.path.exists(dest_path + '.part'):
                os.remove(dest_path + '.part')
        except Exception:
            pass
        return False

# Subprocess helper for pip installs
async def run_subprocess(cmd, timeout=900):
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        out = stdout.decode(errors='ignore') if stdout else ''
        err = stderr.decode(errors='ignore') if stderr else ''
        return proc.returncode, out + ('\n' + err if err else '')
    except Exception as e:
        return 1, str(e)

# Ensure transformers & torch available (used when AUTO_DOWNLOAD true)
async def ensure_transformers_installed():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        pass
    if not AUTO_DOWNLOAD:
        print("تنزيل تلقائي معطّل (AUTO_DOWNLOAD=false)", flush=True)
        return False
    print("AUTO_DOWNLOAD مفعّل: جاري محاولة تثبيت مكتبات transformers و torch...", flush=True)
    cmds = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "transformers", "accelerate", "safetensors"],
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
    ]
    for cmd in cmds:
        code, output = await run_subprocess(cmd, timeout=1200)
        print('تشغيل الأمر:', ' '.join(cmd), 'النتيجة:', code, flush=True)
        print(output[:2000], flush=True)
        if code != 0:
            print('فشل أحد خطوات التثبيت؛ سيتم إيقاف المحاولة التلقائية.', flush=True)
            return False
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception as e:
        print('فشل الاستيراد بعد التثبيت:', e, flush=True)
        return False

async def load_transformers_model():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        print(f"جاري تحميل/تنزيل النموذج {MODEL_NAME} من Hugging Face...", flush=True)
        devnull = open(os.devnull, 'w')
        try:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        finally:
            devnull.close()
        device = 'cuda' if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        if device == 'cuda':
            model.to('cuda')
        else:
            model.to('cpu')
        print('اكتمل تحميل نموذج transformers', flush=True)
        return model, tokenizer, device
    except Exception as e:
        print('فشل تحميل نموذج transformers:', e, flush=True)
        traceback.print_exc()
        return None, None, None

async def generate_with_llama_cpp_prompt(prompt: str, max_tokens: int = 256, temp: float = 0.7) -> str:
    bin_path = LLAMA_CPP_BIN
    model_path = LLAMA_MODEL_PATH
    if not os.path.isfile(bin_path):
        alt = os.path.join('llama.cpp', 'main')
        if os.path.isfile(alt):
            bin_path = alt
        else:
            print('ملف تنفيذ llama.cpp غير موجود', flush=True)
            return None
    if not os.path.isfile(model_path) or os.path.getsize(model_path) < MIN_MODEL_BYTES:
        print('ملف النموذج غير موجود أو غير مكتمل لِـ llama.cpp', flush=True)
        return None
    cmd = [bin_path, '-m', model_path, '--threads', '4', '--temp', str(temp), '--n_predict', str(max_tokens), '--prompt', prompt]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        text = stdout.decode(errors='ignore') if stdout else ''
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
        print('خطأ أثناء تشغيل llama.cpp:', e, flush=True)
        traceback.print_exc()
        return None

async def generate_with_transformers(model, tokenizer, device, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
    try:
        import torch
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if device == 'cuda':
            input_ids = input_ids.to('cuda')
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()
    except Exception as e:
        print('خطأ أثناء توليد الرد من transformers:', e, flush=True)
        traceback.print_exc()
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
    if re.match(r"^[0-9\s\+\-\*\/\. %()]+$", text):
        try:
            r = eval(text, {"__builtins__":None}, {})
            return f"الناتج: {r}"
        except Exception:
            return "ما قدرت أحسب هذا التعبير."
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
