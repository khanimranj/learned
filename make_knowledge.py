from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pickle

directory="pdf_files"
load_dotenv()
text=""


    
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".pdf"):
            print("Extracting from",file)
            pdf_reader = PdfReader(directory+'\\'+file)
            for page in pdf_reader.pages:
                text +=page.extract_text()
                print("Knowledge size",len(text))


    
    # Split
text_splitter= CharacterTextSplitter(separator="\n", chunk_size=500,chunk_overlap=100, length_function=len)
chunks=text_splitter.split_text(text)
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)
with open("knowledge_base.pkl", "wb") as file:
    pickle.dump(knowledge_base, file)
