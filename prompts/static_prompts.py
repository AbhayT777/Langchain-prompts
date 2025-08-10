from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
      repo_id= "mistralai/Mistral-7B-Instruct-v0.3",
      task= "text-generation"
    )

model = ChatHuggingFace(llm=llm)

st.header('Research paper summarization')

user_input = st.text_input('Enter the paper name you want to summarize')

if st.button('summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
