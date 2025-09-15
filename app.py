import streamlit as st
from huggingface_hub import login
from transformers import pipeline
import random
import os

# Hugging Face login (Token should be in Streamlit Secrets)
login(token=st.secrets["HF_HUB_TOKEN"])

# Load emotion detection pipeline
@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=False)

emotion_pipeline = load_model()

# Emoji mapping
emotion_emojis = {
    "anger": "🤬",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😀",
    "neutral": "😐",
    "sadness": "😭",
    "surprise": "😲"
}

# Response generation
def generate_response(emotion):
    responses = {
        "joy": [
            "😊 I'm glad to hear that! Tell me more.",
            "😄 That’s great! What else is going on?",
            "🌟 Awesome! You’re making my day better too!",
        ],
        "sadness": [
            "😔 I'm really sorry you're feeling this way. I'm here for you.",
            "💙 It's tough, but you're not alone.",
            "🌧️ Would you like to talk more about it?",
        ],
        "anger": [
            "😡 That sounds upsetting. I'm listening.",
            "💢 I can sense you're frustrated.",
            "⚡️ What's making you feel this way?",
        ],
        "disgust": [
            "🤢 That doesn’t sound good. Want to tell me more?",
            "😣 That must've been unpleasant.",
        ],
        "fear": [
            "😨 That sounds scary. Are you okay?",
            "🫣 I'm here for you. Want to talk about it?",
        ],
        "surprise": [
            "😲 Whoa! That’s unexpected!",
            "🤯 Sounds like something surprising happened.",
        ],
        "neutral": [
            "🤖 I’m here to chat. What’s on your mind?",
            "💬 How can I help you today?",
        ]
    }
    return random.choice(responses.get(emotion, responses["neutral"]))

# Streamlit UI
st.set_page_config(page_title="Aya Emotion Detection Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>💬 Aya Emotion Detection Chatbot</h1>", unsafe_allow_html=True)
st.markdown("Enter a message, and I’ll detect your emotion and respond like a real chat!")

# Chat messages (stored in session)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="input_text")

if user_input:
    detected_emotion = emotion_pipeline(user_input)[0]['label']
    emoji = emotion_emojis.get(detected_emotion, "🤖")
    response = generate_response(detected_emotion)

    # Add to top of chat
    st.session_state.chat_history.insert(0, {
        "user": user_input,
        "emotion": f"{detected_emotion} {emoji}",
        "bot": response
    })

# Chat display
st.markdown("### 🧠 Recent Messages")
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}  \n> *Emotion:* `{chat['emotion']}`  \n**Aya:** {chat['bot']}")
