Step 1 : Create a private Google API key and store it in the .env file.

Step 2 : Create a requirements.txt file listing all dependency libraries.
            run the below command 
            pip install -r requirements.txt

Step 3 :    1) Create a app.py is a Q & A application to ask the question to gemini-pro LLM to get the accurate answer
            2) Create a vision.py is a Q & A application with image to ask the question to gemini-pro-vision LLM to get the accurate answer
            3) Create a QAChat.py is a Chat Q & A application to ask the question to gemini-pro LLM to get the accurate answer
            4) Create a pdfReaderQAApp.py is a Multi document Q & A application to ask the question to gemini-pro LLM to get the accurate answer

Step 4 :  run the program in terminal,
             > streamlit run app.py
             > streamlit run vision.py
             > streamlit run QAChat.py
             > streamlit run pdfReaderQAApp.py