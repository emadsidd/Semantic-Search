import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
import os
from tqdm import tqdm
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, render_template_string
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

# Configuration
sentence_transformer_path = os.path.join(script_dir, 'Models/all-MiniLM-L6-V2/fine_tuned_model/checkpoint-33876')  # Path to SentenceTransformer
flan_t5_path = os.path.join(script_dir, 'Models/FLAN-T5-small/fine_tuned_model/checkpoint-8675')  # Path to FLAN-T5
faiss_index_path = os.path.join(script_dir, 'Models/all-MiniLM-L6-V2/FAISS/Index/faiss_index_v7.index')  # Path to FAISS index
metadata_path = os.path.join(script_dir, 'Models/all-MiniLM-L6-V2/FAISS/Data/product_metadata_v7.csv')  # Path to metadata CSV

# Database configuration
db_config = {
    'dbname': 'mydb',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load SentenceTransformer model
try:
    sentence_transformer = SentenceTransformer(sentence_transformer_path, device=device)
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    exit(1)

# Load the FLAN-T5 model
try:
    tokenizer = T5Tokenizer.from_pretrained(flan_t5_path)
    flan_t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_path).to(device)
    print("FLAN-T5 model loaded successfully.")
except Exception as e:
    print(f"Error loading FLAN-T5 model: {e}")
    exit(1)

# Load FAISS index
print("Loading FAISS index...")
try:
    index = faiss.read_index(faiss_index_path)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit(1)

# Load the metadata
print("Loading metadata...")
try:
    metadata = pd.read_csv(metadata_path)
    print(f"Metadata shape: {metadata.shape}")
except Exception as e:
    print(f"Error loading metadata: {e}")
    exit(1)

parent_asins = metadata['parent_asin'].tolist()

# Connect to the database
def get_db_connection():
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Cleaning function
def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(text.split())  # Remove extra whitespace
    return text


def extract_filters(query):
    """
    Use the FLAN-T5 model to extract structured filters from the input query.
    """
    input_text = query.strip()

    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).input_ids.to(device)

    # Generate predictions
    outputs = flan_t5_model.generate(input_ids, max_length=96, num_beams=1)

    # Decode the generated output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Fix the output if it's missing enclosing braces
    if not decoded_output.startswith("{") and not decoded_output.endswith("}"):
        decoded_output = "{" + decoded_output + "}"

    # Parse the output as JSON
    try:
        filters = json.loads(decoded_output)
        return filters
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from model output: {decoded_output}")
        return None


# Initialize Flask app
app = Flask(__name__)

# HTML template for rendering results with Bootstrap
html_template = '''
<!doctype html>
<html lang="en">
<head>
    <title>Product Search</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
<div class="container mt-5">
    <!-- Display execution time and step times in the top-right corner -->
    <div class="d-flex justify-content-end">
        <p class="text-muted">
            {% if step_summary %}
                {{ step_summary }}
            {% endif %}
        </p>
    </div>

    <h1 class="mb-4">Cell Phone and Cell Phone Accessories</h1>
    <form method="post" class="mb-5">
        <div class="input-group">
            <input type="text" name="query" class="form-control" placeholder="Enter your search query" required>
            <button class="btn btn-primary" type="submit">Search</button>
        </div>
    </form>

    <!-- Display extracted filters -->
    {% if filters %}
        <h2 class="mb-3">Extracted Filters:</h2>
        <ul class="list-group mb-5">
            <li class="list-group-item"><strong>Price Range:</strong> {{ filters.get('price_min', 'N/A') }} - {{ filters.get('price_max', 'N/A') }}</li>
            <li class="list-group-item"><strong>Review Count Range:</strong> {{ filters.get('review_count_min', 'N/A') }} - {{ filters.get('review_count_max', 'N/A') }}</li>
            <li class="list-group-item"><strong>Average Rating Range:</strong> {{ filters.get('average_rating_min', 'N/A') }} - {{ filters.get('average_rating_max', 'N/A') }}</li>
            <li class="list-group-item"><strong>Subcategory:</strong> {{ filters.get('subcategory', 'N/A') }}</li>
        </ul>
    {% endif %}

    {% if results %}
        <h2 class="mb-3">Results:</h2>
        {% for result in results %}
            <div class="card mb-3">
                <div class="card-body">
                    <h5 class="card-title">{{ loop.index }}. {{ result.title }}</h5>
                    <p class="card-text"><strong>Average Rating:</strong> {{ result.average_rating }}</p>
                    <p class="card-text"><strong>Rating Number:</strong> {{ result.rating_number }}</p>
                    <p class="card-text"><strong>Features:</strong> {{ result.features }}</p>
                    <p class="card-text"><strong>Description:</strong> {{ result.description }}</p>
                    <p class="card-text"><strong>Details:</strong> {{ result.details }}</p>
                    <p class="card-text"><strong>Price:</strong> {{ result.price }}</p>
                    <p class="card-text"><strong>Subcategory:</strong> {{ result.subcategory }}</p>
                </div>
            </div>
        {% endfor %}
    {% endif %}
</div>
<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def search():
    results = []
    execution_time = None
    filters = None
    step_times = {}    
    step_summary = ""      
    
    if request.method == 'POST':
        start_time = time.time()  # Record start time

        # Step 1: Extract Filters using FLAN-T5
        step_start = time.time()
        query = request.form.get('query', '').strip()
        try:
            filters = extract_filters(query)
        except Exception as e:
            print(f"Error extracting filters: {e}")
            filters = {}  # Fallback to empty dictionary if extraction fails
        print(filters)
        print(f"Step 1: FLAN-T5 filter extraction took {time.time() - step_start:.2f} seconds")
        step_times['FLAN-T5 Filter Extraction'] = time.time() - step_start

        # Step 2: Perform semantic search using SentenceTransformer + FAISS
        step_start = time.time()
        cleaned_query = clean_text(query)
        query_embedding = sentence_transformer.encode([cleaned_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = index.search(query_embedding, k=50)
        print(f"Step 2: FAISS search took {time.time() - step_start:.2f} seconds")
        step_times['Semantic Search (FAISS)'] = time.time() - step_start

        # Step 3: Fetch metadata with SQL filtering
        step_start = time.time()
        top_parent_asins = [parent_asins[i] for i in indices[0]]
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        sql_query = '''
            SELECT parent_asin, title, average_rating, rating_number, features, description, details, price, subcategory
            FROM products
            WHERE parent_asin IN %s
            AND (%s IS NULL OR price >= %s)
            AND (%s IS NULL OR price <= %s)
            AND (%s IS NULL OR rating_number >= %s)
            AND (%s IS NULL OR rating_number <= %s)
            AND (%s IS NULL OR average_rating >= %s)
            AND (%s IS NULL OR average_rating <= %s)
            AND (%s IS NULL OR subcategory = %s)
        '''
        try:
            cursor.execute(
                sql_query,
                (
                    tuple(top_parent_asins),
                    filters.get('price_min'), filters.get('price_min'),
                    filters.get('price_max'), filters.get('price_max'),
                    filters.get('review_count_min'), filters.get('review_count_min'),
                    filters.get('review_count_max'), filters.get('review_count_max'),
                    filters.get('average_rating_min'), filters.get('average_rating_min'),
                    filters.get('average_rating_max'), filters.get('average_rating_max'),
                    filters.get('subcategory'), filters.get('subcategory')
                )
            )
            products = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            products = []

        print(f"Step 3: SQL query took {time.time() - step_start:.2f} seconds")
        step_times['SQL Query Execution'] = time.time() - step_start
        
        # Step 4: Prepare results
        step_start = time.time()
        for product in products:
            results.append(product)
        print(f"Step 4: Result preparation took {time.time() - step_start:.2f} seconds")
        step_times['Result Preparation'] = time.time() - step_start
        
        execution_time = time.time() - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")

        step_summary = " | ".join([f"{step}: {time:.2f}s" for step, time in step_times.items()])
        step_summary = f"Total execution time: {execution_time:.2f}s | {step_summary}"

    return render_template_string(html_template, results=results, filters=filters, execution_time=execution_time, step_summary=step_summary)


if __name__ == '__main__':
    app.run(debug=True)