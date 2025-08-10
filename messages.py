from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
      repo_id= "mistralai/Mistral-7B-Instruct-v0.3",
      task= "text-generation"
    )

model = ChatHuggingFace(llm=llm)

msg = [
     SystemMessage(content= 'you are ahelpful assistant'),
     HumanMessage(content= 'tell me about llm chat models')
]

res = model.invoke(msg)

msg.append(AIMessage(content= res.content))
print(msg)