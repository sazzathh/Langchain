from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

load_dotenv()

model=ChatGroq(model="llama-3.1-8b-instant")
chat_history=[SystemMessage(content="you are a helpful assistant.Always answer in 2-4 lines")]
#chatbot
while True:
	user_input=input("you: ")
	chat_history.append(HumanMessage(content=user_input))

	if user_input=="exit":
		break

	result=model.invoke(chat_history)
	chat_history.append(AIMessage(content=result.content))
	print("AI: ",result.content)

print("chat history: ",chat_history)