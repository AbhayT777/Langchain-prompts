from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

llm = HuggingFaceEndpoint(
      repo_id= "mistralai/Mistral-7B-Instruct-v0.3",
      task= "text-generation"
    )

model = ChatHuggingFace(llm=llm)

st.header('Reasearch paper summarization Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })

# if st.button('Summarize'):
#     result = model.invoke(prompt)
#     st.write(result.content)



if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,               #     instead of invoking template and model one by one, we can invoke them in 
        'style_input':style_input,               #     one go,  by using chain and then invoking chain.
        'length_input':length_input
    })
    st.write(result.content)