from flask import Flask, request, render_template
from elasticsearch import Elasticsearch
import psycopg2

app = Flask(__name__)

# Connect to Elasticsearch
es = Elasticsearch("http://localhost:9200", http_auth=("elastic", "your_password"))

# PostgreSQL connection settings
psql_config = {
    "dbname": "mydb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

@app.route("/", methods=["GET", "POST"])
def search():
    query = ""
    results = []

    if request.method == "POST":
        query = request.form.get("query")

        # Elasticsearch query
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "description", "main_category", "features"]
                }
            }
        }

        response = es.search(index="products", body=body, size=20)
        es_results = response["hits"]["hits"]

        # Extract parent_asin values
        parent_asins = [result["_source"]["parent_asin"] for result in es_results]

        # Fetch product details from PostgreSQL
        try:
            conn = psycopg2.connect(**psql_config)
            cursor = conn.cursor()

            # Fetch details for the retrieved parent_asins
            query_placeholders = ", ".join(["%s"] * len(parent_asins))
            sql_query = f"""
                SELECT parent_asin, title, features, description, details, price, subcategory, average_rating, rating_number
                FROM products_table
                WHERE parent_asin IN ({query_placeholders})
            """
            cursor.execute(sql_query, tuple(parent_asins))
            rows = cursor.fetchall()

            # Combine database results into dictionaries
            for row in rows:
                results.append({
                    "parent_asin": row[0],
                    "title": row[1],
                    "features": row[2],
                    "description": row[3],
                    "details": row[4],
                    "price": row[5],
                    "subcategory": row[6],
                    "average_rating": row[7],
                    "rating_number": row[8]
                })

        except Exception as e:
            print(f"Error fetching data from PostgreSQL: {e}")

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # Pass results to template
    return render_template("search.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
