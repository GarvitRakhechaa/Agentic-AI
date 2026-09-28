#do all the imports 

import os 
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, VectorParams, PointStruct
from sentence_transformers import CrossEncoder, SentenceTransformer
from openai import OpenAI
import json
from rank_bm25 import BM25Okapi
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

#creating embeddings  and reranker
emb_model = SentenceTransformer("all-MiniLM-L6-v2") #we are using this because because its dimesions are 384 which we have written in qdrant collection creation
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

# 03 search with threshold
def search_with_filter_and_threshold(query, query_filter=None, threshold_score = 0.9, top_k = 5):
    embedded_query = emb_model.encode(query).tolist() # same for here also 
    results = qd_client.query_points(
        collection_name=Collection_Name,
        query=embedded_query,
        limit= top_k,
        with_payload=True,
        query_filter=query_filter,
        score_threshold=threshold_score
    ).points

    return results

# 04 filter with rerank score
def filter_using_rerank_score(query, results,final_k=3):
    if not results:
        return []

    pairs = [
        [
            query,
            result.payload["text"]
        ] for result in results
    ]

    rerank_scores = reranker.predict(pairs)

    scored_results = []

    for result, score in zip(results, rerank_scores):

        scored_results.append({
            "result":result,
            "rerank_score": float(score)
        })

    scored_results.sort(key= lambda x:x["rerank_score"], reverse=True)

    return scored_results[:final_k]

#05 rerank worker
def rerank_worker(query, top_k=10,):
    results = search(query,top_k)
    return filter_using_rerank_score(query,results)

# 06 Keyword search for hybrid search
def keyword_search(query, top_k=10):

    tokenized_documents = [
        document["text"].lower().split() 
        for document in documents
        ]
    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = query.lower().split()
    keyword_score = bm25.get_scores(query_tokens)

    ranked_results = []
    for i ,score in enumerate(keyword_score):

        ranked_results.append({
            "document_index":i,
            "document":documents[i],
            "keyword_score":float(score)
        })

    ranked_results.sort(key= lambda x:x["keyword_score"], reverse=True)

    return ranked_results[:top_k]

# 07 hybrid search
def hybrid_search(query, top_k=10, final_k=3, vector_weight=0.5, keyword_weight=0.5):
    vector_results = search(query, top_k)

    keyword_result = keyword_search(query, top_k)

    combined_results = {}

    for result in vector_results:

        combined_results[result.id] = {
            "result": result,
            "vector_score": result.score,
            "keyword_score": 0
        }

    for item in  keyword_result:
        qdrant_id = item["document_index"] + 1

        if qdrant_id not in combined_results:
            # retrieve that qdrant documents
            qdrant_result = qd_client.retrieve(
                collection_name=Collection_Name,
                ids=[qdrant_id],
                with_payload= True
            )[0]

            combined_results[qdrant_id] = {
                "result": qdrant_result,
                "vector_score": 0,
                "keyword_score": item["keyword_score"]
            }

        else:
            combined_results[qdrant_id]["keyword_score"] = (
                item["keyword_score"]
            )

        # Vector score normalization
    vector_scores = [
        item["vector_score"]
        for item in combined_results.values()
    ]

    max_vector_score = max(vector_scores) if vector_scores else 1

    # Keyword score normalization
    keyword_scores = [
        item["keyword_score"]
        for item in combined_results.values()
    ]

    max_keyword_score = max(keyword_scores) if keyword_scores else 1

    # Calculate hybrid score
    for item in combined_results.values():

        normalized_vector_score = (
            item["vector_score"] / max_vector_score
            if max_vector_score != 0
            else 0
        )

        normalized_keyword_score = (
            item["keyword_score"] / max_keyword_score
            if max_keyword_score != 0
            else 0
        )

        item["hybrid_score"] = (
            vector_weight * normalized_vector_score
            +
            keyword_weight * normalized_keyword_score
        )

    # Sort by hybrid score
    ranked_results = list(combined_results.values())

    ranked_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return ranked_results[:final_k]


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

def ask_llm(question, search_results_from_vector_db, rerank=False, hybrid=False):

    if not search_results_from_vector_db:
            return "I dont know based on provided information"
        
    context = ""

    for item in search_results_from_vector_db:
        if rerank:
            result = item["result"]
            score = item["rerank_score"]
            print(f"Rerank Score: {score:.3f}")
        elif hybrid:
            result = item["result"]
            vector_score = item["vector_score"]
            keyword_score = item["keyword_score"]
            hybrid_score = item["hybrid_score"]

            print(f"Vector Score: {vector_score:.3f}")
            print(f"Keyword Score: {keyword_score:.3f}")
            print(f"Hybrid Score: {hybrid_score:.3f}")
        else:
            result = item 
            score = item.score
            print(f"Score: {score:.3f}")

        context += result.payload["text"] + "\n"
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





question = """
As a new employee, I want to understand my company policies about leave and reimbursements. How many paid vacation days do I get each year, how far in advance should I submit a vacation request, can unused vacation days be carried forward, can I claim reimbursement for internet expenses used for work, what document is required for the internet claim, and how long does the reimbursement process take?
"""

answer_simple_search = search(question, top_k=3)
answer_filter_query_search = search_with_filter(question,condition_filter)
answer_threshold_search = search_with_filter_and_threshold(question,condition_filter, threshold_score=0.6, top_k=6)
answer_with_rerank_rag = rerank_worker(question)
answer_with_hybrid_search = hybrid_search(question)

final_answer_simple_search = ask_llm(question, answer_simple_search)
final_answer_filter_query_search = ask_llm(question, answer_filter_query_search)
final_answer_threshold_search = ask_llm(question, answer_threshold_search)
final_answer_using_rerank_rag = ask_llm(question,answer_with_rerank_rag,rerank=True)
final_answer_hybrid_search = ask_llm(question,answer_with_hybrid_search, hybrid=True)

print("\nFinal Answer:")
print("answer with simple: ",final_answer_simple_search)
print("answer with filter: ",final_answer_filter_query_search)
print("answer with threshold: ",final_answer_threshold_search)
print("answer with Rerank RAG: ",final_answer_using_rerank_rag)
print("answer with hybrid search: ",final_answer_hybrid_search)