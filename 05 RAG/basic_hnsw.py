# import 
import os
import json 

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, HnswConfigDiff, MatchValue, PointStruct, VectorParams, Filter
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# 2 load env variables
load_dotenv()

Qdrant_Url = os.getenv("CLUSTER_ENDPOINT")
Qdrant_Api_Key = os.getenv("QUADRANT_API_KEY")
Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")

model = "qwen/qwen3.5-plus:free"

# 3 connect Qdrant
qd_client = QdrantClient(
    api_key=Qdrant_Api_Key,
    url=Qdrant_Url
)

print("connected to Qdrant!")


# 4 creating collection
Collection_Name = "filter_knowledge"
EMBEDDING_SIZE = 384

if qd_client.collection_exists(Collection_Name):
    qd_client.delete_collection(Collection_Name)

qd_client.create_collection(
    collection_name=Collection_Name,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(
            m=16, #showing connectivity of graph
            ef_construct=100 # connectivity makes better because this can explore larger candidate pool
        )
    )
)

print("collection created: ", Collection_Name)

#  document loading
with open("./05 RAG/knowledge2.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"loaded {len(documents)} documents")  

# embedding model load
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# making embeddings of text only
texts = [document["text"] for document in documents]

docs_embeddings = embed_model.encode(
    texts,
    show_progress_bar=True 
    )

print("Embeddings generated")
print("Embedding size:", len(docs_embeddings[0]))

# creating qdrant points 
points = []

# making points where id and vector is vector embedding of that text of that document and payload is it self document
for i , doc in enumerate(documents):
    point = PointStruct(
        id = i+1,
        vector=docs_embeddings[i].tolist(),
        payload= doc
    ).point 

    points.append(point)

#upload to udrant 
qd_client.upsert(
    collection_name=Collection_Name,
    points=points
)

print(
    f"Uploaded {len(points)} documents to Qdrant!"
)

def search(query, query_filter = None ,top_k=5):
    query_embedding = embed_model.encode(
        query,
        show_progress_bar=True
    ).tolist()

    results = qd_client.query_points(
        collection_name=Collection_Name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
        search_params={
            "hnsw_ef":128
        }
    ).points

    return results
 
# made filter
reimnursement_filter = Filter(
    must = [
        FieldCondition(
            key="category",
            match=MatchValue(
                value="reimbursement"
            )
        )
    ]
)





