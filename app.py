#Libraries to be imported
import os
import requests
import streamlit as st
from streamlit_lottie import st_lottie

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
st.set_page_config(page_title="Chatbot", page_icon=":desktop_computer:", layout="wide")

#---lottie load funtion---

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#---Load assets---
lottie_asset = load_lottieurl("https://lottie.host/edfd046e-38c7-4e08-997f-0d74f9ed3252/vHppEor6Zm.json")

with st.sidebar:
    #---header section---
    with st.container():
        st.title("Hi there :wave:")
        st.title("Welcome to Vishwajeet's website")
        st.header("Who is Vishwajeet? :thinking_face:")
        st.subheader("Vishwajeet is an experienced Data Scientist with a diverse background spanning ten years, including six years of expertise in machine learning and computer vision projects. Prior to transitioning to data science, he spent four years as a Mechanical Design Engineer, specializing in aerospace and railway part design. Proficient in Python and SQL, he possess a comprehensive understanding of various supervised and unsupervised machine learning techniques, coupled with a proven track record of delivering impactful solutions across various industries.")
        st.write("If you want to know more about Vishwajeet, just ask Yoda. He is right there on right side of the screen")

    #with st.container():

        #input_text = st.text_input("Ask ChatGPT about Vishwajeet's experience or professional work")

        #if input_text:
            #response = retrival_chain.invoke({'input':input_text})
            #st.write(response['answer'])
    
    with st.container():
        st.write("---")
        st.subheader("Follow Vishwajeet on:")
        st.write("[LinkedIn >](https://www.linkedin.com/in/sawantvishwajeet729/)")
        st.write("[Medium >](https://medium.com/@sawantvishwajeet729)")
        st.write("[Email >](sawantvishawjeet729@gmail.com)")

with st.container():
    text_1, anime = st.columns((2, 1))
    with text_1:
        st.header("Hi there :wave:")
        st.write("I am Yoda, Vishwajeet's AI assistant.")
        st.write("You can ask about Vishwajeet's professional work, skills and qualities")
    with anime:
        #st.write('yoda animation')
        st_lottie(lottie_asset, height=50)



with st.container():
    messages = st.container(height=350)
    if input_text := st.chat_input("Ask Yoda about Vishwajeet"):
        messages.chat_message("user").write(input_text)
        response = retrival_chain.invoke({'input':input_text})
        messages.chat_message("ai").write(f"Yoda: {response['answer']}")


