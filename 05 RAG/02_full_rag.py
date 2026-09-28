import os 
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import sys

load_dotenv()
Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")


client = OpenAI(
    api_key=Api_key,
    base_url=Base_url
)

#embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# knowledge base
documents = [
    "Employees receive 24 days of paid leave per year.",
   
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
   
    "Employees receive Rs 3000 per month for gym reimbursement.",
   
    "Employees can claim Rs 2000 per month for home internet.",
   
    "Employees have a 90 day notice period."
]

# document embeddings 
doc_embedding = embedding_model.encode(documents)
print(doc_embedding)
print(sys.getsizeof(doc_embedding))

def cosine_similarity(a, b):
     return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def retrieve(query_embeddings):
    scores = []
    for i, doc in enumerate(doc_embedding):
        print(i)
        score = cosine_similarity(doc, query_embeddings)
        scores.append((score, documents[i])) # adding score with that text line
    scores.sort(reverse=True)
    return scores[0]

def ask_llm(question, context):
    system_prompt = f"""
    answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}
    """

    response = client.chat.completions.create(
        model="qwen/qwen3.5-plus:free",
         messages=[
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user",
                "content":question
            }
         ],
         temperature=0
    )
    return response.choices[0].message.content


query = "How much vacation do I get?"

query_embeddings = embedding_model.encode(query)
score,context = retrieve(query_embeddings=query_embeddings)

answer = ask_llm(query,context)
print(answer)
