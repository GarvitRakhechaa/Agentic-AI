import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")


client = OpenAI(
    api_key=Api_key,
    base_url=Base_url
)

#1 knowledge base
knowledge_base = {
    "age":"the age of gorav is 24",
    "net worth":" network of gorav is 30k usd"
}

# step 2 retrievel 

def retrieve_info(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["net worth"]

def ask_llm(question):
    context = retrieve_info(question)
    sys_prompt = f"""
answer in one line only. Answer only based on this context. do not hallucinate. Context: {context} 
"""
    response = client.chat.completions.create(
    model="qwen/qwen3.5-plus:free",
    messages= [
        {
            "role":"system",
            "content":sys_prompt
        },
        {
            "role":"user",
            "content":question
        }
    ],
    )
    
    print(response.choices[0].message.content)

ask_llm("gorav ki age kitni hai?")