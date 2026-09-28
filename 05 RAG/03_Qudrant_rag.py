# 01 imports
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# 02 load env variables
load_dotenv()

Quadrant_Url = os.getenv("CLUSTER_ENDPOINT")
Quadrant_Api_Key = os.getenv("QUADRANT_API_KEY")

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")


# 03 create Qd_client
client = QdrantClient(
    api_key=Quadrant_Api_Key,
    url=Quadrant_Url
)

print("connected to Qdrant")

Collection_name = "knowledge"
Embedding_size = 384


# 04 create collection
if client.collection_exists(Collection_name):
    print(f"deleting existing collection: {Collection_name}")
    client.delete_collection(Collection_name)

client.create_collection(
  collection_name=Collection_name,
  vectors_config= VectorParams(
      size=Embedding_size,
      distance=Distance.COSINE
  )  
)

print(f"Created collection: {Collection_name}")
print(f"Vector size: {Embedding_size}")
print("Distance: COSINE")

# 05 load the knowledgebase
with open('./05 RAG/knowledge.txt',"r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip() ]
    print(f"document loaded: {len(documents)} documents")



#06 creating embeddings 
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2") #384
print("Embedding model ready!")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")


# 07 creating qdrant_points
points = []
for i , embedding in enumerate(embeddings):
    point = PointStruct(
        id=i+1,
        vector = embedding.tolist(),
        payload = {
            "text": documents[i]
        }
    )
    points.append(point)


#08 upload to qdrant
client.upsert(
    collection_name= Collection_name,
    points=points
)

print(f"Uploaded {len(points)} documents to Qdrant!")



#09 simple_search from qdrant
def simple_search(query, top_k=3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=Collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points
    return results

openai_client = OpenAI(
    api_key=Api_key,
    base_url=Base_url
)


def ask_llm(question, context):

    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't know based on the provided information."
"""

    response = openai_client.chat.completions.create(
        model="qwen/qwen3.5-plus:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

question = "How many vacation days do I get?"

results = simple_search(question, top_k=3)

context = "\n".join(result.payload["text"] for result in results )

answer = ask_llm(question,context)


print("\nFinal Answer:")
print(answer)