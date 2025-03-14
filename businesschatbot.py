import os
import json
import ssl
import random
import streamlit as st
import nltk
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fix SSL issues for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# ✅ Load intents file
file_path = "updated_green_intents.json"
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if "intents" in data:
            intents = data["intents"]
        else:
            raise ValueError("Invalid JSON structure: Missing 'intents' key.")
except Exception as e:
    st.error(f"Error loading JSON file: {e}")
    intents = []

# ✅ Preprocess training data
patterns, responses, tags = [], [], []
for intent in intents:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        responses.append(random.choice(intent["responses"]))
        tags.append(intent["tag"])

# ✅ Train TF-IDF Model
vectorizer = TfidfVectorizer()
if patterns:
    x_train = vectorizer.fit_transform(patterns)
else:
    st.error("No patterns found in intents. Check your JSON file.")

# ✅ Function to log chat history in CSV format (but NOT display it)
def log_chat(user_input, bot_response):
    log_file = "chat_log.csv"
    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User Input": user_input,
        "Chatbot Response": bot_response
    }
    
    # Append new chat logs to the CSV file
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(log_file, index=False)

# ✅ Chatbot response function using Cosine Similarity
def chatbot(input_text):
    if not patterns:
        return "I am currently unavailable due to a system error."

    input_vec = vectorizer.transform([input_text])
    similarity_scores = cosine_similarity(input_vec, x_train).flatten()

    best_match_index = np.argmax(similarity_scores)
    confidence = similarity_scores[best_match_index]

    # ✅ Confidence threshold (ignores irrelevant inputs)
    if confidence < 0.3:
        return "I'm sorry, I don't understand. Can you rephrase?"

    return responses[best_match_index]

# ✅ Streamlit Chatbot UI (No Chat Log Display)
def main():
    st.title("🌱 Green Business Consultant Chatbot")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar Menu
    menu = ["Chat", "Assessment", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Chat":
        st.write("Welcome! Type a message below and press Enter to chat.")

        # Display previous chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["text"])

        # User Input Field (Keeps the conversation open)
        user_input = st.chat_input("Type your message...")

        if user_input:
            # Display user message
            st.session_state.chat_history.append({"role": "user", "text": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate and display chatbot response
            bot_response = chatbot(user_input)
            st.session_state.chat_history.append({"role": "assistant", "text": bot_response})
            with st.chat_message("assistant"):
                st.write(bot_response)

            # ✅ Log chat history in CSV file (No Display)
            log_chat(user_input, bot_response)

    elif choice == "Assessment":
        st.subheader("Green Business Assessment")
        run_assessment()

    elif choice == "Conversation History":
        st.header("Conversation History")
        for msg in st.session_state.chat_history:
            st.text(f"{msg['role'].title()}: {msg['text']}")

    elif choice == "About":
        st.subheader("About the Green Business Consultant Chatbot")
        st.write("""
        This chatbot helps businesses assess their sustainability practices.
        It can:
        - Answer common questions about green business practices.
        - Run a **Green Business Assessment** to evaluate sustainability.
        - Provide insights based on the assessment score.
        """)

# ✅ Green Business Assessment
def run_assessment():
    responses = {}

    questions = [
        ("business_name", "What is the name of your business?"),
        ("industry", "What industry does your business operate in?"),
        ("employees", "How many employees work in your company? (Enter a number)"),
        ("energy_source", "What is your primary energy source? (Renewable, Fossil fuels, Mixed)"),
        ("waste_recycling", "Do you use a waste recycling system? (Yes/No)"),
        ("recycling_percentage", "What percentage of waste is recycled? (e.g., 20, 50, 80)"),
        ("water_conservation", "Do you implement water conservation measures? (Yes/No)"),
        ("carbon_tracking", "Do you track your carbon emissions? (Yes/No)"),
        ("green_certifications", "Do you hold any green certifications? (Yes/No)"),
        ("sustainability_goals", "Are you working towards any sustainability goals? (e.g., Net Zero, Carbon Neutral)"),
    ]

    for key, prompt in questions:
        response = st.text_input(prompt, key=key)
        if response.lower() in ["quit", "exit"]:
            st.warning("Assessment canceled.")
            return
        responses[key] = response

    st.write("**Summary of Your Responses:**")
    for key, value in responses.items():
        st.text(f"{key.replace('_', ' ').title()}: {value}")

    confirm = st.button("Submit Assessment")
    if confirm:
        score = calculate_score(responses)
        category = categorize_business(score)
        st.success(f"Assessment Complete! Your sustainability score is **{score}**.")
        st.info(f"Your business is categorized as: **{category}**.")

# ✅ Score Calculation Function
def calculate_score(responses):
    score = 0
    if responses.get("energy_source", "").lower() == "renewable":
        score += 20
    if responses.get("waste_recycling", "").lower() == "yes":
        score += 10
    if responses.get("water_conservation", "").lower() == "yes":
        score += 10
    if responses.get("carbon_tracking", "").lower() == "yes":
        score += 10
    if responses.get("green_certifications", "").lower() == "yes":
        score += 20
    return score

# ✅ Business Categorization Function
def categorize_business(score):
    if score >= 50:
        return "Sustainable Business"
    elif score >= 30:
        return "Moderately Sustainable"
    else:
        return "Needs Improvement"

if __name__ == "__main__":
    main()
