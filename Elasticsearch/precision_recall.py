import json
from elasticsearch import Elasticsearch
from elasticsearch import helpers

# Load ground truth
with open('ground_truth.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)

# Connect to Elasticsearch
es = Elasticsearch(
    ["http://127.0.0.1:9200"],
    basic_auth=("username", "password")  # Replace with your credentials
)

index_name = "products_index"  

def search_es(query, es, index, size=20):
    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "description", "features", "details", "subcategory"]
            }
        },
        "size": size
    }
    response = es.search(index=index, body=body)
    hits = response["hits"]["hits"]
    return [hit["_source"]["parent_asin"] for hit in hits if "_source" in hit and "parent_asin" in hit["_source"]]

def precision_recall_at_k(relevant_asins, retrieved_asins, k=10):
    relevant_set = set(relevant_asins)
    retrieved_set = set(retrieved_asins[:k])

    relevant_retrieved = relevant_set.intersection(retrieved_set)
    precision = len(relevant_retrieved) / float(k) if k > 0 else 0.0
    recall = len(relevant_retrieved) / float(len(relevant_set)) if len(relevant_set) > 0 else 0.0
    return precision, recall

# Automate retrieval
retrieval_results = {}
for query in ground_truth.keys():
    # Automatically query Elasticsearch
    retrieved_asins = search_es(query, es, index_name, size=10)
    retrieval_results[query] = retrieved_asins

# Compute Precision@10 and Recall@10
results = {}
precision_list = []
recall_list = []

for query, relevant_asins in ground_truth.items():
    retrieved_asins = retrieval_results.get(query, [])
    p, r = precision_recall_at_k(relevant_asins, retrieved_asins, k=10)
    results[query] = {
        "Precision@10": p,
        "Recall@10": r
    }
    precision_list.append(p)
    recall_list.append(r)

# Print the results for each query
for query, metrics in results.items():
    print(f"Query: {query}")
    print(f"  Precision@10: {metrics['Precision@10']:.4f}")
    print(f"  Recall@10: {metrics['Recall@10']:.4f}")
    print("-" * 30)

# Compute and print average precision and recall
avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0
avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0

print("Average Precision@10: {:.4f}".format(avg_precision))
print("Average Recall@10: {:.4f}".format(avg_recall))
