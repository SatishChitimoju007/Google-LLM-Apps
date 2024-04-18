from dotenv import load_dotenv
load_dotenv()
from PIL import Image

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# for vision images - gemini pro vision

model = genai.GenerativeModel("gemini-pro-vision")

def get_gemini_respone(input, image):
    if input != "":
        response = model.generate_content([input,image])
    else:
        response = model.generate_content(image)
    return response.text

st.set_page_config(page_title="Q&A Demo")
st.header("Gemini LLM application")

input = st.text_input("Input: ", key="input")
upload_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

image = ""
if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption="Upload image:", use_column_width=True)

submit = st.button("Ask the question")

if submit:
    response = get_gemini_respone(input, image)
    st.subheader("The respone is")
    st.write(response)