import os
import pandas as pd
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


script_dir = os.path.dirname(os.path.abspath(__file__))

def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII
    text = text.lower()  # Convert to lowercase
    text = ' '.join(text.split())  # Remove extra whitespace
    return text


product_embeddings_path = os.path.join(script_dir, 'Models/all-MiniLM-L6-V2/Data/sample_product_embeddings/sample_product_embeddings.npy')
processed_products_path = os.path.join(script_dir, 'Models/all-MiniLM-L6-V2/Data/sample_product_embeddings/sample_product_metadata.csv')
original_data_path = os.path.join(script_dir, 'Models/all-MiniLM-L6-V2/Data/evaluation_sample_10000.csv')

# Load precomputed product embeddings
product_embeddings = np.load(product_embeddings_path)

# Load product metadata
df_product_metadata = pd.read_csv(processed_products_path)

# Load original dataset 
df_original = pd.read_csv(original_data_path)


# Extract relevant queries and clean them
df_queries = df_original[df_original['query'] != 'irrelevant']
queries = df_queries['query'].unique()
cleaned_queries = [clean_text(query) for query in queries]

# Initialize Sentence Transformer and generate query embeddings
model_path = os.path.join(script_dir, 'Models/all-MiniLM-L6-V2/fine_tuned_model/checkpoint-33876')
model = SentenceTransformer(model_path)
query_embeddings = model.encode(cleaned_queries, batch_size=64, show_progress_bar=True)

# Map queries to their relevant product IDs
query_to_relevant = df_queries.groupby('query')['parent_asin'].apply(set).to_dict()


k = 20
precision_list = []
recall_list = []

print("\nEvaluating Precision@k and Recall@k...")
for i, query in enumerate(cleaned_queries):
    print(f"Query: {query}")

    # Compute cosine similarity between query and product embeddings
    query_embedding = query_embeddings[i].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, product_embeddings).flatten()

    # Get top-k product indices and details
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_products = df_product_metadata.iloc[top_k_indices]['parent_asin'].values

    # Get ground truth relevant products
    relevant_products = query_to_relevant[queries[i]]
    
    # Check retrieved relevance
    relevant_retrieved = 0
    print(f"{'Rank':<5} {'ASIN':<15} {'Similarity':<10} {'Relevant'}")
    for rank, asin in enumerate(top_k_products, start=1):
        similarity = similarities[top_k_indices[rank - 1]]
        is_relevant = asin in relevant_products
        relevant_retrieved += 1 if is_relevant else 0
        print(f"{rank:<5} {asin:<15} {similarity:.4f} {'Yes' if is_relevant else 'No'}")

    # Calculate Precision@k and Recall@k
    precision_at_k = relevant_retrieved / k
    recall_at_k = relevant_retrieved / len(relevant_products)
    
    # Append to lists for overall metrics
    precision_list.append(precision_at_k)
    recall_list.append(recall_at_k)
    
    # Print metrics for the query
    print("\nMetrics:")
    print(f"Precision@{k}: {precision_at_k:.4f}")
    print(f"Recall@{k}: {recall_at_k:.4f}")
    print(f"Total Relevant: {len(relevant_products)}, Relevant Retrieved: {relevant_retrieved}\n")


mean_precision = np.mean(precision_list)
mean_recall = np.mean(recall_list)

print(f"Overall Metrics:")
print(f"Mean Precision@{k}: {mean_precision:.4f}")
print(f"Mean Recall@{k}: {mean_recall:.4f}")
