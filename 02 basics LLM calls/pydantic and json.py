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

from pydantic import BaseModel
class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema = Ticket.model_json_schema()

response_format = {
    "type":"json_object"
}

system_prompt=f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""

text="Hello My name is Pratyush. Yesterday I broke up with my girlfriend sheetal I have an iphone which is not working at all. My address is delhi. My email is abc@gmail.com. My contact number is 82134"
prompt=f"""
This is a customer ticket. Please extract the personal information from this.
{text}
"""

response = client.chat.completions.create(
    model="qwen/qwen3.5-plus:free",
    messages=[
        {
            "role":"system",
            "content": system_prompt
        },
        {
            "role":"user",
            "content":prompt
        }
    ],
    response_format=response_format
)

answer = response.choices[0].message.content
print(answer)

import json 
data = json.loads(answer)
ticket = Ticket(**data)


print(ticket.name)
print(ticket.email)
print(ticket.issue)