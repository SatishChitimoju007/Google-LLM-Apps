import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configure GenAI with the Google API key obtained from the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# get_pdf_text(pdf_docs): 
# Create a PdfReader object to read the PDF document
# Iterate through each page of the PDF document
# Extract text from the current page and append it to the text string
# Return the concatenated text extracted from all PDF documents
def get_pdf_text(pdf_docs): 
    text=""
    for pdf in pdf_docs: #read all the pages in the pdf
        pdf_reader=PdfReader(pdf) #with the help of PDfreader we are reading it will readit
        for page in pdf_reader.pages:
            text+=page.extract_text()# we are extracting the info from pages
    return text

# get_text_chunks(text):
# Converting into chunks
# Initialize a RecursiveCharacterTextSplitter with specified chunk size and overlap
# Split the text into chunks using the text splitter
# Return the list of text chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

# get_vector_store(text_chunks):
# getting the vectors using google embedding
# Initialize Google's Generative AI Embeddings model with the specified path
# Create a vector store using FAISS from the given text chunks and embeddings
# Save the vector store locally with the name "faiss_index"
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    vector_store =FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

# get_conversational_chain():
# Define a template for the prompt with placeholders for context and question
# Initialize a Google Generative AI model for conversation
# Create a prompt template using the defined template and input variables
# Load a question-answering chain using the specified model, chain type, and prompt
# Return the conversational question-answering chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide the 
    context just say, "answer is not available in the context", don't provide the 
    context: \n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User_input(user_question):
# Initialize Google's Generative AI Embeddings model with the specified path
# Load the vector store locally using FAISS, allowing dangerous deserialization
# Perform similarity search to retrieve relevant documents for the user question
# Get a conversational question-answering chain
# Generate a response using the conversational chain
# Pass input documents and the user question to the chain
# Specify to return only the output text
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain =get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])

# main():
# Input field for the user to ask a question from the PDF files
# Call the user_input function to handle the user's question
# File uploader for uploading multiple PDFs
# Extract text from the uploaded PDFs
# Split the extracted text into chunks
# Create a vector store from the text chunks
# Display a success message once processing is complete
def main():
    st.set_page_config("Chat with multiple PDF")
    st.header("Chat with multiple pdfs using Gemini LLM ")

    user_question = st.text_input("Ask a question from the pdf files ")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Multiple documents:")
        pdf_docs = st.file_uploader("upload the pdfs", accept_multiple_files=True)
        if st.button("submit and process"):
            with st.spinner("processing...."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Uploaded successfully..!")

if __name__ == "__main__":
    main()