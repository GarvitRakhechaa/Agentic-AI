from email import message
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")

client = OpenAI(
    api_key=Api_key,
    base_url=Base_url
)


def ask_llm(prompt):
    response = client.chat.completions.create(
        model="qwen/qwen3.5-plus:free",
        messages=[
            {
                "role":"user",
                "content":prompt 
            }
        ],   
    )
    answer = response.choices[0].message.content
    print(answer)


prompt = """ 
You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue in a category
#CONSTRAINT
You have to classify the issue in one of three categories namely billing, technical, return.
#OUTPUT FORMAT
Your answer should be in one word only. The one word shoud be one of the categories given in constraints
#Example
For instance if a user compalin says he wants a refund then the category is Return
#FALLBACK
If the issue is unrelated to any of the categories mentioned in constraints, then the answer should be OTHER
This is a user complaint:
my marriage is broke
"""

ask_llm(prompt)