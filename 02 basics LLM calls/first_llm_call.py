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
            "role":"user",
            "content":"hi, this is my first API call"
        }
    ],
)

print(response.choices[0].message.content)