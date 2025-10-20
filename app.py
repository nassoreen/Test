from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

# ✅ อนุญาตให้ React Native หรือเว็บแอปเรียก API นี้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือกำหนด origin ของแอปคุณ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 โหลดโมเดล Keras (.keras)
model = load_model("Flowermodel8.keras")

# 🌸 Class names (แก้ให้ตรงกับตอน train)
class_names = [
    "เดซี่","แดนดิไลอัน","ชบา","ดอกเข็ม","มะลิ","บัว","ดาวเรือง","กล้วยไม้","กุหลาบ","ทานตะวัน","ทิวลิป", "ไม่ทราบ"
]


# 📐 ขนาด input ของโมเดล (ตามที่ train ไว้)
IMG_SIZE = 224

# ✅ หน้า root
@app.get("/")
def root():
    return {"message": "🌸 Flower Classifier API (Keras) is running!"}
    
# 📷 Predict Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # อ่านไฟล์ภาพ
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # แปลงเป็น array และ preprocess
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ทำนาย
    predictions = model.predict(img_array)
    probs = tf.nn.softmax(predictions[0]).numpy()

    pred_index = np.argmax(probs)
    pred_class = class_names[pred_index]
    confidence = float(np.max(probs))

    return {
        "class_index": int(pred_index),
        "class_name": pred_class,
        "confidence": confidence,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }
