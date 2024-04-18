from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# for text - gemini pro

model = genai.GenerativeModel("gemini-pro")

def get_gemini_respone(question):
    response = model.generate_content(question)
    return response.text

st.set_page_config(page_title="Q&A Demo")
st.header("Gemini Text LLM application")

input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit:
    response = get_gemini_respone(input)
    st.subheader("The response is")
    st.write(response)
    