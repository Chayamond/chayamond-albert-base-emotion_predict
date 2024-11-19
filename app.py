import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# โหลดโมเดลและ tokenizer จาก Hugging Face
model_name = "chayamond/albert-base-emotion_predict"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/albert-base-v2-emotion")

# สร้างฟังก์ชันสำหรับการทำนาย
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)  # คำนวณ softmax เพื่อให้ได้ความน่าจะเป็น
    return predictions


st.title("Text Emotional Detector")
user_input = st.text_area("Enter the Text:")

if st.button("Predict"):
    if user_input:
        predictions = predict_emotions(user_input)
        class_labels = ["joy", "fear", "anger", "sadness"]  # แก้ไขตามคลาสของโมเดล
        predictions = predictions.detach().numpy()[0]  # แปลง Tensor เป็น numpy array

        # ดึงค่าที่มีความมั่นใจสูงสุด
        max_index = predictions.argmax()  # ตำแหน่งของค่าที่มากที่สุด
        predicted_emotion = class_labels[max_index]
        confidence = predictions[max_index]

        # แสดงผลลัพธ์
        st.write(f"### Predicted Emotion: **{predicted_emotion}**")
        st.write(f"Confidence: {confidence:.2f}")
