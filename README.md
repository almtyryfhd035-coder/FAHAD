# FAHAD - Discord AI Bot

هذا المشروع يجهّز بوت دسكورد يستخدم واجهة Hugging Face Inference للردود باللغة العربية (اللهجة السعودية العامية).

المحتويات:
- bot.py: كود البوت
- Dockerfile: لبناء الصورة على Render
- render.yaml: (اختياري) ملف إعداد خدمة Render
- requirements.txt: تبعيات المشروع

إعداد وتشغيل على Render (خطوات سريعة):
1. ادخل إلى https://dashboard.render.com وابدأ خدمة جديدة (New -> Web Service).
2. اختر GitHub ووصّل مستودع almtyryfhd035-coder/FAHAD.
3. اختر Environment: Docker
4. إذا طُلب Dockerfile path اتركه كما هو (Dockerfile).
5. في Environment > Environment Variables أضف المتغيرات التالية:
   - DISCORD_BOT_TOKEN = توكن البوت (احصل عليه من بوابة Discord Developer Portal)
   - HF_API_TOKEN = توكن Hugging Face (من https://huggingface.co/settings/tokens)
   - MODEL_NAME = اسم الموديل على Hugging Face (مثال: gpt2 أو موديل حواري أفضل إن وُجد)
6. قم بنشر الخدمة.

ملاحظات وأفضل الممارسات:
- أنصح باستخدام موديل حوار مُصمّم للردود، وليس gpt2 الافتراضي؛ ابحث في Hugging Face عن نماذج دردشة متوافقة مع Inference API.
- لا تُشارك مفاتيحك (DISCORD_BOT_TOKEN أو HF_API_TOKEN) في المحادثات العامة.
- إن رغبت أن أضبط لك اسم الموديل الافتراضي أو أضيف تحسينات إضافية (تخزين سياق المحادثة، حدود الطول، فلترة)، أخبرني وأطبقها.

