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


st.title("Phishing Email Detector")
user_input = st.text_area("Enter the email content:")

st.write(predict_phishing(user_input))
if st.button("Predict"):
    if user_input:
        predictions = predict_emotions(user_input)
        class_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]  # แก้ไขตามคลาสของโมเดล
        predictions = predictions.detach().numpy()[0]  # แปลง Tensor เป็น numpy array

        st.write("### Predicted Emotions:")
        for label, score in zip(class_labels, predictions):
            st.write(f"**{label}:** {score:.2f}")
