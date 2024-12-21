import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch, helpers

# Load the deduplicated dataset
df = pd.read_csv('testing_sample_10000_unique.csv', encoding='utf-8-sig')

# Replace NaN with None so that it becomes valid JSON null
# This will ensure Elasticsearch doesn't choke on NaN.
df = df.where(pd.notnull(df), None)

# Convert to a list of dicts
documents = df.to_dict(orient='records')

es = Elasticsearch(
    ["http://127.0.0.1:9200"],
    basic_auth=("username", "password")  # Replace with your credentials
)

index_name = "products_index"

# Delete index if it exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

mapping = {
    "mappings": {
        "properties": {
            "parent_asin": {"type": "keyword"},
            "title": {"type": "text"},
            "features": {"type": "text"},
            "description": {"type": "text"},
            "subcategory": {"type": "text"},
            "details": {"type": "object", "enabled": False}
        }
    }
}
es.indices.create(index=index_name, body=mapping)

actions = [
    {
        "_index": index_name,
        "_id": doc["parent_asin"],
        "_source": doc
    }
    for doc in documents
]

try:
    helpers.bulk(es, actions)
    print("Indexing completed successfully.")
except helpers.BulkIndexError as bulk_error:
    print("Bulk indexing error occurred:")
    for error in bulk_error.errors:
        print(error)
