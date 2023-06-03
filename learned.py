#from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from PIL import Image
import os
import pickle
#load_dotenv()
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]
with open("knowledge_base.pkl", "rb") as file:
    knowledge_base = pickle.load(file)
llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff")



def main():
    st.set_page_config(page_title="IMD Weatherman")
    st.header("Interactive weather Chat")
    logo_image = Image.open("logo.jpg")

    # Display the logo in the top left-hand side
    st.image(logo_image, use_column_width=False, width=100)
    
    # upload file
    
    # extract the text

      # show user input
    user_question = st.text_input("Ask me a question about what I learned:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
        
        st.write(response)
    

if __name__ == '__main__':
    main()
