# Furniture Web Model Backend

Flask backend service for furniture recommendation and related product discovery.

This project powers room-based furniture suggestions by loading product data from MySQL, normalizing product features, and returning recommendation results through REST APIs.

## Features

- Room-based furniture recommendations (`/api/search`)
- Product details by ID (`/api/product/<product_id>`)
- ML-based related products (`/api/product/<product_id>/related`)
- Available filter metadata (`/api/filters`)
- Health check endpoint (`/api/health`)

## Tech Stack

- Python 3.10+
- Flask + Flask-CORS
- Pandas + NumPy
- scikit-learn (OneHotEncoder, cosine similarity)
- SQLAlchemy + PyMySQL

## Project Structure

- `app.py` - API server and endpoint definitions
- `model.py` - data loading, preprocessing, and recommendation logic
- `evaluate_model.py` - evaluation script (currently based on older dataset-driven flow)
- `requirements.txt` - Python dependencies
- `Dataset/` - local dataset files (legacy/evaluation)

## Prerequisites

Before running the backend, make sure you have:

1. Python installed
2. MySQL server running
3. A database with required tables used by `model.py`:
   - `products` (`id`, `name`, `description`, `price`, `category_id`, `is_deleted`)
   - `categories` (`id`, `name`, `is_deleted`)
   - `product_features` (`product_id`, `feature_name`, `feature_value`, `is_deleted`)
   - `product_images` (`product_id`, `image_path`, `createdAt`, `is_deleted`)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Update the database credentials in `app.py`:

```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'thilina_ecommerce',
    'port': 3306,
    'charset': 'utf8mb4',
    'autocommit': True
}
```

## Run the Server

```bash
python app.py
```

Default server address:

- `http://localhost:5001`

## API Endpoints

### 1) Search Recommendations

- **Method:** `POST`
- **Endpoint:** `/api/search`

Request body:

```json
{
  "roomSize": "small",
  "roomType": "living-room",
  "style": "modern",
  "budget": "low",
  "limit": 5
}
```

Notes:

- `roomType` supports values like `living-room` (converted internally to `Living Room`)
- strict matching is used on room size, room type, style, and budget range
- `limit` is optional (default: `5`)

Sample response:

```json
{
  "title": "Modern Living Room Design",
  "description": "Exact-match recommendations for your small living room in modern style (low budget).",
  "suggestions": [
    {
      "id": "12",
      "name": "Minimal Sofa",
      "price": 89900.0,
      "description": "Compact 2-seater sofa",
      "image": "images/sofa.jpg",
      "link": "/product/12",
      "match_score": 100.0
    }
  ],
  "tips": ["..."],
  "tags": ["Small", "Living Room", "Modern", "Low Budget"]
}
```

### 2) Get Product by ID

- **Method:** `GET`
- **Endpoint:** `/api/product/<product_id>`

Sample response:

```json
{
  "product": {
    "id": "12",
    "name": "Minimal Sofa",
    "price": 89900.0
  }
}
```

### 3) Get Related Products

- **Method:** `GET`
- **Endpoint:** `/api/product/<product_id>/related?limit=10`

Notes:

- returns top related items based on cosine similarity
- max limit is `10`

Sample response:

```json
{
  "product_id": "12",
  "related_products": [
    {
      "id": "34",
      "name": "Modern Coffee Table",
      "price": 49900.0,
      "description": "...",
      "image": "...",
      "link": "/product/34",
      "match_score": 87.42
    }
  ]
}
```

### 4) Get Available Filters

- **Method:** `GET`
- **Endpoint:** `/api/filters`

Returns:

- room sizes
- room types
- styles
- budget ranges
- minimum and maximum price

### 5) Health Check

- **Method:** `GET`
- **Endpoint:** `/api/health`

Sample response:

```json
{
  "status": "healthy",
  "message": "Room Planner API is running"
}
```

## Common Error Responses

- `400` - missing required fields in request
- `404` - product not found / no related products found
- `500` - service unavailable or internal server error

## Logging

Logs are written to:

- console output
- `app.log`

## Notes

- `model.py` normalizes feature-name variations (e.g., `Room Size`, `room size`, `ROOM_SIZE` -> `room_size`).
- `evaluate_model.py` uses an older approach and may need refactoring to work directly with the current DB-backed model constructor.
