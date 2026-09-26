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

JD="""
We are hiring a Backend Python Developer.

Requirements:
- Strong Python
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""

RESUME="""
Name: Rahul Sharma

Experience:
3 years as a Software Developer.

Skills:
Python, FastAPI, MySQL, Docker,
REST APIs, Git

Projects:
Built a food delivery backend using
FastAPI and MySQL.

Deployed applications using Docker.
"""

def ask_llm(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="qwen/qwen3.5-plus:free",
        messages=[
            {
                "role":"system",
                "content": system_prompt
            },
            {
                "role":"user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content 


def step_1_res_extract(RESUME):
    print("STEP 1")
    system_prompt="""
    You are a professional HR assistant. Extract the skills from the candidates resume provided.
    Only return the skills no other information. Do not invent any skillsby yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt=f"""
    Extract the skills from this resume
    {RESUME}
    """

    return ask_llm(system_prompt, user_prompt)

def step_2_JD_extract(JD):
    print("step2")
    system_prompt="""
    You are a professional HR assistant. Extract the skills from the Job description  provided.
    Only return the skills no other information. Do not invent any skills by yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt=f"""
    Extract the skills from this JD
    {JD}
    """

    return ask_llm(system_prompt, user_prompt)

def step_3_match(candidate,jd):
    print("step3")
    system_prompt="""
    You are a professional HR assistant. compare the skills of candidate and the skills required in the JD and produce a final score between
    1 and 100. also produce a short verdict whther the candidate is a good fit for the role.
    """
    user_prompt=f"""
    Compare and matc h the skills
    JD:
    {jd}
    Candidate:
    {candidate}
    """

    return ask_llm(system_prompt, user_prompt)

candidate = step_1_res_extract(RESUME)
print("candidate: ", candidate)
print("\n")

jd= step_2_JD_extract(JD)
print("job description: ", jd)

score = step_3_match(candidate, jd)

print("score: ", score)