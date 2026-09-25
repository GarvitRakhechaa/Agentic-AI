import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")


client = OpenAI(
    api_key=Api_key,
    base_url=Base_url
)

response = client.chat.completions.create(
    model="qwen/qwen3.5-plus:free",
    messages= [
        {
            "role":"system",
            "content":"you are my best friend"  # this will be role of llm now
        },
        {
            "role":"user",
            "content":"hi, this is my 2nd API call appreciate me "
        }
    ],
    temperature=0, # 0 creativity
    max_completion_tokens=500 # max output tokens
)

print(response.choices[0].message.content)