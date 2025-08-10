from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = HuggingFaceEndpoint(
      repo_id= "mistralai/Mistral-7B-Instruct-v0.3",
      task= "text-generation"
    )

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content= 'you are a helpful assistant'),
    HumanMessage(content= 'tell me about transformers in bert model')
]

while True:
    user_input = input("you: ")
    chat_history.append(HumanMessage(content= user_input))
    if user_input == "exit":
        break
    res = model.invoke(chat_history)
    chat_history.append(AIMessage(content= res.content))
    print("AI: ", res.content)

print(chat_history)
