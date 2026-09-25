from dotenv import load_dotenv
import os 
from openai import OpenAI
load_dotenv()


Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")

client = OpenAI(
    api_key=Api_key,
    base_url=Base_url
)


response_format={
    "type" : "json_object"
}

def API_CALL(user_prompt, system_prompt=""):

    messages = []
    if system_prompt:
        messages.append({
            "role":"system",
            "content":system_prompt
        })

    messages.append({
        "role":"user",
        "content":user_prompt
    })

    response = client.chat.completions.create(
        model="qwen/qwen3.5-plus:free",
        messages=messages,
        response_format=response_format
    )
    return response.choices[0].message.content