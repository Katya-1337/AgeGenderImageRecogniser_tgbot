import os
import random
import tempfile
import logging
from typing import Tuple, Dict, Any

import numpy as np
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ---------- CONFIG ----------
AGE_MODEL_PATH = "./agemodel.keras"      # или .h5
GENDER_MODEL_PATH = "./genmodel.keras"  # или .h5

# Порядок индексов для гендер-модели: 0=male, 1=female
GENDER_LABELS = ["male", "female"]

# Лимит размера изображения перед подачей (на всякий случай)
MAX_IMAGE_SIDE = 2048
# Фиксированный размер для инференса
INFERENCE_IMAGE_SIZE = 200

# Фразы пока бот обрабатывает фото (случайный выбор)
THINKING_PHRASES = [
    "Не торопись, я думаю...",
    "Не туплю, а думаю...",
    "Не скроль мемы, сейчас всё будет.",
    "Ща, нейросеть грузится...",
    "Считаю пиксели, не мешай.",
    "Мозги кипят, подожди секунду.",
    "Не завис — просто умный.",
    "Думаю. Да, это редкость.",
    "Секунду, нейроны шевелю...",
    "Гружу мозги с диска...",
    "Обрабатываю... кофе бы.",
    "Думаю. Даже сам в шоке.",
    "Подожди, нейросети тоже устают.",
    "Ща по нейросетевому расписанию.",
]
# ----------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError("Не найден TELEGRAM_TOKEN в переменных окружения (.env).")


def load_models():
    age_model = tf.keras.models.load_model(AGE_MODEL_PATH)
    gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH)
    return age_model, gender_model


AGE_MODEL, GENDER_MODEL = load_models()


def get_target_size(model: tf.keras.Model) -> Tuple[int, int]:
    """
    Берем входной размер модели (H, W).
    Ожидается формат [None, H, W, C].
    """
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    # input_shape: (None, H, W, C)
    h, w = input_shape[1], input_shape[2]
    if h is None or w is None:
        # fallback если динамический размер
        return 200, 200
    return int(w), int(h)


def preprocess_image_for_model(image: Image.Image, model: tf.keras.Model) -> np.ndarray:
    """
    Универсальная предобработка:
    - RGB
    - resize под вход модели
    - float32 [0..1]
    - batch dimension
    """
    image = image.convert("RGB")

    # ограничим сверхбольшие картинки
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)

    target_w, target_h = get_target_size(model)
    image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr


def format_classification_prediction(pred: np.ndarray, labels=None) -> str:
    """
    Форматирует предсказание для:
    - binary output (shape [1,1] или [1])
    - multiclass (shape [1,num_classes])
    """
    pred = np.array(pred)

    # Уберем batch размерность
    if pred.ndim >= 2 and pred.shape[0] == 1:
        pred = pred[0]

    # binary
    if pred.ndim == 0:
        value = float(pred)
        return f"value={value:.6f}"

    if pred.ndim == 1:
        if pred.shape[0] == 1:
            p = float(pred[0])  # вероятность female
            cls = "female" if p >= 0.5 else "male"
            cls_ru = "женский" if cls == "female" else "мужской"
            percent = (p * 100) if cls == "female" else ((1 - p) * 100)
            return f"Предполагаемый пол: {cls_ru}, уверенность модели: {percent:.0f}%"

        # multiclass
        best_idx = int(np.argmax(pred))
        best_prob = float(pred[best_idx])
        if labels and 0 <= best_idx < len(labels):
            name = labels[best_idx]
        else:
            name = str(best_idx)

        top_k = min(3, len(pred))
        top_idx = np.argsort(pred)[-top_k:][::-1]
        top_text = []
        for i in top_idx:
            i = int(i)
            label = labels[i] if labels and i < len(labels) else str(i)
            top_text.append(f"{label}: {float(pred[i]):.4f}")

        return (
            f"top1={name} ({best_prob:.4f})\n"
            f"top{top_k}: " + ", ".join(top_text)
        )

    return f"raw={pred.tolist()}"


def format_age_prediction(pred: np.ndarray) -> str:
    """
    Форматирует регрессионный выход agemodel как возраст.
    """
    pred = np.array(pred)
    if pred.ndim >= 2 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim == 0:
        age = float(pred)
    elif pred.ndim == 1 and pred.shape[0] >= 1:
        age = float(pred[0])
    else:
        return f"raw={pred.tolist()}"

    return f"Предполагаемый возраст: {age:.0f} лет"


def run_models(image: Image.Image) -> Dict[str, Any]:
    x_age = preprocess_image_for_model(image, AGE_MODEL)
    x_gender = preprocess_image_for_model(image, GENDER_MODEL)

    # Сначала agemodel (регрессия), затем genmodel (классификация male/female).
    y_age = AGE_MODEL.predict(x_age, verbose=0)
    y_gender = GENDER_MODEL.predict(x_gender, verbose=0)

    return {
        "agemodel": format_age_prediction(y_age),
        "genmodel": format_classification_prediction(y_gender, GENDER_LABELS),
    }


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь фото, и я проанализирую картинку и верну возраст и пол."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Просто отправь изображение (как фото или файл) — проанализирую картинку и верну возраст и пол."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Если отправлено как фото:
        if update.message.photo:
            file = await update.message.photo[-1].get_file()
        # Если отправлено как документ:
        elif update.message.document and update.message.document.mime_type and update.message.document.mime_type.startswith("image/"):
            file = await update.message.document.get_file()
        else:
            await update.message.reply_text("Пожалуйста, отправь изображение.")
            return

        await update.message.reply_text(random.choice(THINKING_PHRASES))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name

        await file.download_to_drive(custom_path=tmp_path)

        image = Image.open(tmp_path)
        results = run_models(image)

        text = (
            "Результаты:\n\n"
            f"{results['agemodel']}\n\n"
            f"{results['genmodel']}"
        )
        await update.message.reply_text(text)

    except Exception as e:
        logger.exception("Ошибка при обработке изображения")
        await update.message.reply_text(f"Ошибка обработки: {e}")
    finally:
        # Удалим временный файл
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ответ на текстовые сообщения: бот работает только с фотографиями."""
    await update.message.reply_text(
        "Я умею работать только с фотографиями. Загрузите, пожалуйста, изображение."
    )


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))

    # Принимаем фото и image-документы
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
    # Обычные сообщения — просим отправить фото
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling()


if __name__ == "__main__":
    main()