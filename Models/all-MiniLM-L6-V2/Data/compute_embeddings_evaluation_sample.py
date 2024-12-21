import os
import pandas as pd
import numpy as np
import json
import re
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


script_dir = os.path.dirname(os.path.abspath(__file__))

def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(text.split())  # Remove extra whitespace
    return text


def preprocess_text(row):
    title = clean_text(row['title']) if pd.notnull(row['title']) else ''
    features = clean_text(' '.join(literal_eval(row['features']))) if pd.notnull(row['features']) else ''
    description = clean_text(' '.join(literal_eval(row['description']))) if pd.notnull(row['description']) else ''
    details_raw = row['details'] if pd.notnull(row['details']) else ''
    try:
        details_dict = json.loads(details_raw.replace("'", '"'))
        details = clean_text(' '.join([f"{k}: {v}" for k, v in details_dict.items()]))
    except json.JSONDecodeError:
        details = ''
    combined_text = f"{title} {features} {description} {details}".strip()
    return combined_text


# Create a folder if it doesn't exist
def check_folder_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)


# Load dataset
data_path = os.path.join(script_dir, 'Data/evaluation_sample_10000.csv')
df = pd.read_csv(data_path)

# Drop duplicate products based on unique attributes
df_unique_products = df.drop_duplicates(subset=['parent_asin', 'title', 'features', 'description', 'details']).copy()

# Preprocess product text
print("Cleaning and preprocessing product text...")
df_unique_products['processed_text'] = df_unique_products.apply(preprocess_text, axis=1)

# Initialize Sentence Transformer model
model_path = os.path.join(script_dir, 'fine_tuned_model/checkpoint-33876')
model = SentenceTransformer(model_path)

# Generate embeddings for unique products
print("Generating embeddings for products...")
product_embeddings = model.encode(df_unique_products['processed_text'].tolist(), batch_size=16, show_progress_bar=True)

# Ensure the output folders exist
check_folder_exists(os.path.join(script_dir, 'Data/sample_product_embeddings'))

# Save product embeddings
product_embeddings_path = os.path.join(script_dir, 'Data/sample_product_embeddings/sample_product_embeddings.npy')
np.save(product_embeddings_path, product_embeddings)

# Save only the parent_asin to map with embeddings
processed_data_path = os.path.join(script_dir, 'Data/sample_product_embeddings/sample_product_metadata.csv')
df_unique_products[['parent_asin']].to_csv(processed_data_path, index=False)

print(f"Product embeddings saved to {product_embeddings_path}")
print(f"Processed parent_asins saved to {processed_data_path}")
