#Rajan Hans - Jan 08 2024
import streamlit as streamlit
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


load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# one or more pdf files can be dropped into the files sections, 
# for each pdf file read each page add to the text string
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# split the complete text string into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    chunks=text_splitter.split_text(text)
    return chunks

# create vector embeddings of each of the chunks
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    
# initialize and create the chain using the prompt template and gemini-pro model 
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:"""
    model =ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])

    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


#this is  the question and answer section. First load the local database and 
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question":user_question}, return_only_outputs=True)

    print(response)
    streamlit.write("Reply:", response["output_text"])


def main():
    streamlit.set_page_config("Rajan's Chat with Multiple PDF")
    streamlit.header("Rajan's Multi-PDF Chat feat. Gemini💁")

    user_question = streamlit.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with streamlit.sidebar:
        streamlit.title("Menu:")
        pdf_docs = streamlit.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if streamlit.button("Submit & Process"):
            with streamlit.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                streamlit.success("Done")



if __name__ == "__main__":
    main()

    

                   
    






