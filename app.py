#Libraries to be imported
import os
import requests
import streamlit as st
#from streamlit_lottie import st_lottie

#---Langchain Libraries---
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI


#---Env variables---
os.environ['LANGCHAIN_TRACING_V2'] = 'False'
os.environ['OPENAI_API_KEY']=st.secrets["openAIKey"]


#---Set-up the ChatGPT---
#load vector db
db = FAISS.load_local("v_resume_vec", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

#ChatGPT model
model=ChatOpenAI()

#create prompt template
prompt = ChatPromptTemplate.from_template("""
                                          Ypu are an assistant and should answer the following question based only on the context provided. 
                                          <context>
                                          {context}
                                          </context>
                                          Question: {input}""")

#create document chain
document_chain = create_stuff_documents_chain(model, prompt)

#convert database to retriver
retriver = db.as_retriever()

#create retrical chain
retrival_chain = create_retrieval_chain(retriver, document_chain)

#---Streamlit code ---
st.set_page_config(page_title="Chatbot", page_icon=":desktop_computer:")

#---lottie load funtion---

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#---Load assets---
#lottie_asset = load_lottieurl("https://lottie.host/6c5c7896-f364-402a-8384-959f8c1d57da/Vz1FkDM4Uj.json")

#---header section---
with st.container():
    st.subheader("Hi there :wave:")
    st.title("Welcome to Vishwajeet's AI chatbot")

    st.subheader("I am here to help you get to know Vishwajeet's professional experience")

with st.container():
    st.write("---")
    st.subheader("ChatGPT")
    text_col, image_col = st.columns((2,1))

    with text_col:
        input_text = st.text_input("Ask ChatGPT about Vishwajeet's experince or professional work")

    with text_col:
        if input_text:
            response = retrival_chain.invoke({'input':input_text})
            st.write(response['answer'])
        #st_lottie(lottie_asset)


with st.container():
    st.write("---")
    st.subheader("Follow Vishwajeet on:")
    st.write("[LinkedIn >](https://www.linkedin.com/in/sawantvishwajeet729/)")
    st.write("[Medium >](https://medium.com/@sawantvishwajeet729)")
    st.write("[Email >](sawantvishawjeet729@gmail.com)")
