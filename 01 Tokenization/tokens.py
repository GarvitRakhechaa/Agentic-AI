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

prompt1 = "Hi!"
prompt2 = "Explain time travel in Detail but under 100 words"
prompt3 = "Write a 1000 word essay on Machine learning"

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {
        "role":"user",
        "content":prompt 
    }

    messages = [message]

    response = client.chat.completions.create(
    model="qwen/qwen3.5-plus:free",
    messages=messages,
        
    temperature=0, # 0 creativity
    max_completion_tokens=5000 # max output tokens
    )
    usage = response.usage 
    print(f" Prompt: {prompt}\n response: {response.choices[0].message.content}\n your tokens: {usage.prompt_tokens} \n completion_tokens: {usage.completion_tokens} \n total tokens: {usage.total_tokens} \n Finish Reason: {response.choices[0].finish_reason}")
    