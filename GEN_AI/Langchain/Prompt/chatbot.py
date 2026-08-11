from langchain_groq import ChatGroq
from dotenv import load_dotenv

import streamlit as st

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# st.header('Chatbot')

while True:
    
    
    user = input("User : ")
    
    if user == 'exit':
        break
    
    chatbot = model.invoke(user)
    
    print(chatbot.content)
    


