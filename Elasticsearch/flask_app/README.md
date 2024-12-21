# Flask Application with Elasticsearch and PostgreSQL Integration

This repository contains a Flask web application that integrates Elasticsearch for product search and PostgreSQL for retrieving detailed product information. The app allows users to search for products, displays relevant filters, and fetches complete product details dynamically.

## Features
- **Search Products**: Users can search for products based on titles, descriptions, or features.
- **Dynamic Results**: Fetch product details from PostgreSQL once Elasticsearch determines relevant product IDs.
- **Responsive UI**: Built with Bootstrap for a modern and responsive design.

---

## Prerequisites

1. **Python**: Ensure Python 3.7+ is installed.
2. **Elasticsearch**: Install and set up Elasticsearch 7.x or later.
3. **PostgreSQL**: Set up a PostgreSQL database containing product data.
4. **Python Libraries**: Install required Python libraries using pip.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/flask_app.git
cd flask_app
```

### 2. Install Dependencies

Create a virtual environment and install the required libraries:

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Elasticsearch

1. Start your Elasticsearch instance and ensure it is running at `http://localhost:9200`.
2. Update `app.py` with your Elasticsearch credentials:
   ```python
   es = Elasticsearch("http://localhost:9200", http_auth=("elastic", "your_password"))
   ```

### 4. Configure PostgreSQL

1. Set up a PostgreSQL database with a table containing product details. Ensure it includes the following columns:
   - `parent_asin`
   - `title`
   - `features`
   - `description`
   - `details`
   - `price`
   - `subcategory`
   - `average_rating`
   - `rating_number`

2. Update the PostgreSQL connection settings in `app.py`:
   ```python
   psql_config = {
       "dbname": "your_database",
       "user": "your_username",
       "password": "your_password",
       "host": "your_host",
       "port": "your_port"
   }
   ```

### 5. Set Up the Directory Structure
Ensure the following structure:

```
flask_app/
│
├── app.py               # Main Flask application
├── templates/           # Folder for HTML templates
│   └── search.html      # HTML template for the web app
├── requirements.txt     # List of Python dependencies
├── README.md            # This README file
```

### 6. Run the Application

Start the Flask app:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`.

---

## Usage

1. Open `http://127.0.0.1:5000` in your web browser.
2. Enter a search query (e.g., "cell phone") in the search bar.
3. View the search results, filters, and detailed product information.

---

## Additional Notes

- Ensure Elasticsearch and PostgreSQL services are running before starting the application.
- The app dynamically fetches product details from PostgreSQL based on `parent_asin` values returned by Elasticsearch.

---


