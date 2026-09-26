import os
from pathlib import Path
from pydoc import resolve
from time import sleep
from dotenv import load_dotenv
import re 
import os 
from openai import OpenAI
import requests
load_dotenv()

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")

client = OpenAI(
    api_key=Api_key,
    base_url=Base_url 
)


def get_product_price(product):
    if product == 'iPhone 17':
        return 1000
    elif product == "iPhone 15":
        return 500
    else:
        return 

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "calc error!"

def get_weather(city):
    url = f"https://wttr.in/{city}?format=%C+%t"
    return requests.get(url)

tools = {
    "get_product_price":get_product_price,
    "calculator":calculator,
    "get_weather": get_weather
}

system_prompt = """
You are a shopping assistant.

You have these tools:

get_product_price(product)
calculator(expression)
IMPORTANT:
Call tools exactly like these examples:

Action: get_product_price("iPhone 17")
Action: calculator("5000 - 1000")
Action: get_weather("jaipur")

Never write:
get_product_price(product="iPhone 17")

Never write:
calculator(expression="5000 - 1000")

Never write:
get_weather(city="Noida")
Follow these rules:

1. Decide what you need to do next.
2. Call ONLY ONE tool at a time.
3. After writing an Action, STOP immediately.
4. Never guess or invent a tool result strictly no invent and guess and valid for observation also dont try to generate price theere is a price tool.
5. Wait until you receive an Observation.
6. Then decide your next action.
7. When the task is complete, give the Final Answer.

Format:

Thought: what you need to do
Action: tool_name(argument)

When finished:

Final Answer: your answer 
"""


def run_agent(question):
    messages = [
        {
            "role":"system",
            "content":system_prompt
        },
        {
            "role":"user",
            "content":question
        }
    ]
    step = 0
    while True:
        print("\n------------------")
        print("STEP", step + 1)
        print("------------------")
        step = step + 1
        response = client.chat.completions.create(
            model="qwen/qwen3.5-plus:free",
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content
        print(answer)

        if "final answer" in answer.lower():
            break 

        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2).strip().strip('""')

            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)

            else:
                observation = "Tool not found"

            print("Observation: ",observation)

            messages.append({
                "role":"assistant",
                "content":"Observation: " + str(observation)
            })


prompt="""
I have 5000 rupees. What is the price of an iphone 17?
and how much money will I have left?
also i want to buy this from jaitaran btw whats the weather of jaitaran 
"""
run_agent(prompt)