#do all the imports 

import os 
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
# load env varialbes at the file 
load_dotenv()

Qdrant_Url = os.getenv("CLUSTER_ENDPOINT")
Qdrant_Api_Key = os.getenv("QUADRANT_API_KEY")
Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")


qd_client =QdrantClient(
    api_key=Qdrant_Api_Key,
    url=Qdrant_Url
)


print("connected!")

Collection_Name = "filter_knowledge"
EMBEDDING_SIZE = 384

if qd_client.collection_exists(Collection_Name):
    qd_client.delete_collection(Collection_Name)

qd_client.create_collection(
    Collection_Name,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)

print(f"Created collection: {Collection_Name}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

# creating index so we can prioritize the category from whole knowledge base 
qd_client.create_payload_index(
    collection_name=Collection_Name,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

#loading knowledge 

with open("./05 RAG/knowledge2.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

#creating embeddings 
emb_model = SentenceTransformer("all-MiniLM-L6-v2") #we are using this because because its dimesions are 384 which we have written in qdrant collection creation

texts = [document["text"] for document in documents] # this will create list of text from loaded json file

embeddings = emb_model.encode(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# creating qdrant points

points = []

for i in range(len(documents)):
    point = PointStruct(
        id=i+1,
        vector=embeddings[i].tolist(),   # we are doing tolist() embedding usually numpy array hoti hai and qdrant expects python list
        payload= documents[i] # we are putting all text, category, is_active as ith object
    )
    points.append(point)

# uploading to Qdrant

qd_client.upsert(
    collection_name=Collection_Name,
    points=points 
)


print(f"Uploaded {len(points)} documents to Qdrant!")
# 01 simple search in RAG
def search(query, top_k = 3):
    embedded_query = emb_model.encode(query).tolist() # same for here also 
    results = qd_client.query_points(
        collection_name=Collection_Name,
        query=embedded_query,
        limit= top_k,
        with_payload=True 
    ).points

    return results

#02 Query filter in RAG
def search_with_filter(query, query_filter=None, top_k = 3):
    embedded_query = emb_model.encode(query).tolist() # same for here also 
    results = qd_client.query_points(
        collection_name=Collection_Name,
        query=embedded_query,
        limit= top_k,
        with_payload=True,
        query_filter=query_filter
    ).points

    return results


condition_filter = Filter(
    must=[
        FieldCondition(
            key="category", # isi basis per filter check hoga
            match= MatchValue(value = "reimbursement")
        )
    ]
)

client = OpenAI(
    api_key=Api_key,
    base_url=Base_url
)

def ask_llm(question, search_results_from_vector_db):
    context = ""

    for result in search_results_from_vector_db:
        context += result.payload["text"] + "\n"
        print(f"Score: {result.score:.3f}")
        print(result.payload["text"])
        print()


    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = client.chat.completions.create(
        model="qwen/qwen3.5-plus:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


question = "What expenses can employees claim for reimbursement, what documents are required for these claims, and how long does the reimbursement process usually take?"

answer_simple_search = search(question, top_k=3)
answer_filter_query_search = search_with_filter(question,condition_filter)



final_answer_simple_search = ask_llm(question, answer_simple_search)
final_answer_filter_query_search = ask_llm(question, answer_filter_query_search)


print("\nFinal Answer:")
print("answer without filter: ",final_answer_simple_search)
print("answer with filter: ",final_answer_filter_query_search)