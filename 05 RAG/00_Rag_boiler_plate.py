import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv()
from openai import OpenAI

Quadrant_Url = os.getenv("CLUSTER_ENDPOINT")
Quadrant_Api_Key = os.getenv("QUADRANT_API_KEY")

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")

client = QdrantClient(
    api_key=Quadrant_Api_Key,
    url=Quadrant_Url
)

print("connected to Qdrant")

Collection_name = "knowledge"
Embedding_size = 384

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
