from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# model name
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Invoke a response
response = model.invoke("Explain quantum computing in two sentences.")
print(response.content)