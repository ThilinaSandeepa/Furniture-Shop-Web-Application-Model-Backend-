# Room Planner AI — Furniture Recommendation Backend

A Flask-based REST API backend that powers an intelligent furniture recommendation engine for a room planner web application. The service loads product data from a MySQL database, builds a machine learning feature space, and returns ranked furniture suggestions and related products through a clean REST interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Search Logic & Outcomes](#search-logic--outcomes)
- [ML Model Details](#ml-model-details)
- [Model Evaluation](#model-evaluation)
- [Logging](#logging)
- [Error Responses](#error-responses)

---

## Overview

This backend receives user preferences (room size, room type, style, budget) and returns a ranked list of matching furniture products. When no exact match exists, the engine falls back to a relaxed search while keeping style and budget constraints strict. A separate ML-powered endpoint returns related products using cosine similarity across categorical and price features.

---

## Features

- **Smart furniture search** — exact-match first, then graceful fallback
- **ML-based related products** — cosine similarity on one-hot encoded attributes plus normalized price
- **Dynamic filter metadata** — returns all valid filter values from the live database
- **Design tips & tags** — contextual design advice generated per query
- **Health check** — simple liveness probe for the service
- **Structured logging** — logs to both console and `app.log`
- **Thread-safe lazy model initialization** — model is loaded once and reused across requests

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Web framework | Flask 2.3, Flask-CORS 4.0 |
| Database ORM | SQLAlchemy 2.0, PyMySQL 1.4 |
| Data processing | Pandas 2.3, NumPy 1.26 |
| Machine learning | scikit-learn 1.5 (OneHotEncoder, cosine similarity) |
| Evaluation charts | Matplotlib |
| Runtime | Python 3.10+ |

---

## Project Structure

```
.
├── app.py                               # Flask application, route definitions, lazy model init
├── model.py                             # RoomPlannerModel — data loading, preprocessing, ML logic
├── evaluate_model.py                    # Offline evaluation script (Hit@K, MRR, Avg Price Gap)
├── requirements.txt                     # Python dependencies
├── search_outcomes_logic.txt            # Plain-text explanation of the 3 search outcomes
├── app.log                              # Runtime log file (auto-created)
├── evaluation_metrics.png               # Output chart from evaluate_model.py (auto-created)
└── Dataset/
    ├── dataset.xlsx                     # Reference dataset for offline evaluation
    └── dataset_backup_before_overwrite.xlsx
```

---

## Database Schema

The model reads from four tables in MySQL. All tables use a soft-delete column `is_deleted`.

### `products`

| Column | Type | Notes |
|---|---|---|
| `id` | INT | Primary key |
| `name` | VARCHAR | Product name |
| `description` | TEXT | Product description |
| `price` | DECIMAL | Product price |
| `category_id` | INT | Foreign key → `categories.id` |
| `is_deleted` | TINYINT | Soft delete flag |

### `categories`

| Column | Type | Notes |
|---|---|---|
| `id` | INT | Primary key |
| `name` | VARCHAR | Used as the base `room_type` value |
| `is_deleted` | TINYINT | Soft delete flag |

### `product_features`

| Column | Type | Notes |
|---|---|---|
| `product_id` | INT | Foreign key → `products.id` |
| `feature_name` | VARCHAR | e.g. `room_size`, `style`, `room_type` (many name variants supported) |
| `feature_value` | VARCHAR | e.g. `Small`, `Modern`, `Living Room` |
| `is_deleted` | TINYINT | Soft delete flag |

### `product_images`

| Column | Type | Notes |
|---|---|---|
| `product_id` | INT | Foreign key → `products.id` |
| `image_path` | VARCHAR | Relative path to product image |
| `createdAt` | DATETIME | Used to pick the earliest (primary) image |
| `is_deleted` | TINYINT | Soft delete flag |

> **Feature name normalization** — the model automatically resolves many casing and spacing variants of feature names to their canonical form. Supported variants include `room_size`, `Room Size`, `RoomSize`, `ROOM_SIZE`, and others. Unrecognized feature names are silently dropped.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Furniture-Web-Model-Backend
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Update the database credentials block at the top of `app.py`:

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

---

## Running the Server

```bash
python app.py
```

The API will be available at:

```
http://localhost:5001
```

The model is initialized lazily on the first request to any endpoint, so the first call may take slightly longer.

---

## API Endpoints

### `POST /api/search` — Search Recommendations

Returns ranked furniture recommendations based on room preferences.

**Request body:**

```json
{
  "roomSize": "Small",
  "roomType": "Living Room",
  "style": "Modern",
  "budget": "Low",
  "limit": 5
}
```

| Field | Required | Description |
|---|---|---|
| `roomSize` | Yes | `Small`, `Medium`, or `Large` |
| `roomType` | Yes | `Living Room`, `Bedroom`, or `Dining` |
| `style` | Yes | `Modern`, `Classic`, or `Minimalist` |
| `budget` | Yes | `Low`, `Medium`, or `High` |
| `limit` | No | Number of results to return (default: `5`) |

**Budget range mapping (derived from product price):**

| Label | Price Range |
|---|---|
| Low | ≤ LKR 80,000 |
| Medium | LKR 80,001 – 200,000 |
| High | > LKR 200,000 |

**Sample response:**

```json
{
  "title": "Modern Living Room Design",
  "description": "Exact-match recommendations for your small living room in modern style (low budget).",
  "suggestions": [
    {
      "id": "12",
      "name": "Minimal Sofa",
      "price": 75000.0,
      "description": "Compact 2-seater sofa in modern style",
      "image": "images/sofa.jpg",
      "link": "/product/12",
      "match_score": 100.0
    }
  ],
  "room_size_adjusted": false,
  "tips": [
    "Choose one hero piece with hidden storage",
    "Pick low-profile furniture and one sculptural chair"
  ],
  "tags": ["Small", "Living Room", "Modern", "Low Budget", "Contemporary", "Sleek", "Cozy", "Entertainment"]
}
```

---

### `GET /api/product/<product_id>` — Get Product by ID

Returns full details for a single product.

**Sample response:**

```json
{
  "product": {
    "id": "12",
    "name": "Minimal Sofa",
    "price": 75000.0,
    "description": "Compact 2-seater sofa in modern style",
    "room_size": "Small",
    "room_type": "Living Room",
    "style": "Modern",
    "budget_range": "Low",
    "image": "images/sofa.jpg"
  }
}
```

---

### `GET /api/product/<product_id>/related` — ML-Based Related Products

Returns up to 10 products most similar to the given product, ranked by cosine similarity.

**Query parameter:**

| Parameter | Default | Max |
|---|---|---|
| `limit` | `10` | `10` |

**Sample response:**

```json
{
  "product_id": "12",
  "related_products": [
    {
      "id": "34",
      "name": "Modern Coffee Table",
      "price": 49900.0,
      "description": "Tempered glass top with steel legs",
      "image": "images/coffee_table.jpg",
      "link": "/product/34",
      "match_score": 87.42
    }
  ]
}
```

---

### `GET /api/filters` — Available Filter Options

Returns all distinct filter values currently present in the database, along with the price range.

**Sample response:**

```json
{
  "room_sizes": ["Large", "Medium", "Small"],
  "room_types": ["Bedroom", "Dining", "Living Room"],
  "styles": ["Classic", "Minimalist", "Modern"],
  "budget_ranges": ["High", "Low", "Medium"],
  "price_range": {
    "min": 12500.0,
    "max": 450000.0
  }
}
```

---

### `GET /api/health` — Health Check

**Sample response:**

```json
{
  "status": "healthy",
  "message": "Room Planner API is running"
}
```

Returns `"status": "degraded"` if the model failed to initialize.

---

## Search Logic & Outcomes

The search endpoint produces one of three outcomes depending on how well the query matches the product catalog.

### Outcome 1 — Exact match found

All four criteria (room size, room type, style, budget) match exactly. Each result receives `match_score: 100.0`.

> Response description: *"Exact-match recommendations for your small living room in modern style (low budget)."*

### Outcome 2 — Fallback match (room size relaxed)

No exact four-field match exists. The engine keeps `room_type`, `style`, and `budget` strict but relaxes `room_size`. Products are ranked by the number of matching fields and then by ascending price. `room_size_adjusted: true` is returned in the response.

> Response description: *"No products found for your [size] … Try adjusting your room size or style to see more options."*

### Outcome 3 — No results

The fallback also yields nothing because no products in the database match `room_type + style + budget` for any room size. `suggestions` is an empty array.

**Key rule:** `style` and `budget` are **never** relaxed. Only `room_size` can be relaxed in the fallback step.

---

## ML Model Details

### Feature Engineering (`model.py`)

1. **Categorical encoding** — `room_size`, `room_type`, `style`, and `budget_range` are one-hot encoded using `sklearn.preprocessing.OneHotEncoder`.
2. **Price normalization** — each product's price is divided by the catalog maximum and repeated across three dimensions.
3. **Combined feature matrix** — one-hot encoded categorical features are horizontally stacked with the price features.

### Related Products — Cosine Similarity

The similarity score for a candidate product is computed as a weighted combination:

```
score = 0.8 × content_similarity + 0.2 × price_similarity
```

where:
- `content_similarity` = cosine similarity between the one-hot encoded categorical vectors
- `price_similarity` = `1 - |price_target - price_candidate|` (on the normalized 0–1 scale)

Products sharing the same `room_type` as the target are ranked above products from other room types, even if their overall score is slightly lower.

### Room Type Inference

If a product's `room_type` is `Unknown` or the category name is not a recognized room-type label, the model infers it from the category name using a keyword map:

| Keywords | Inferred Room Type |
|---|---|
| sofa, couch, loveseat, armchair, coffee table, tv unit, bookshelf, area rug … | Living Room |
| bed, bedroom, wardrobe, dresser, nightstand, mattress … | Bedroom |
| dining, dining table, dining chair, buffet, sideboard … | Dining |

---

## Model Evaluation

`evaluate_model.py` performs leave-one-out evaluation against `Dataset/dataset.xlsx` and reports three metrics:

| Metric | Description |
|---|---|
| **Hit@K** | Fraction of queries where at least one relevant item appears in the top-K results (K = 5) |
| **MRR** | Mean Reciprocal Rank — average of `1 / rank_of_first_relevant_result` |
| **Avg Price Gap** | Mean absolute difference between the query product price and the average price of relevant items |

Run the evaluation:

```bash
python evaluate_model.py
```

A bar chart is saved to `evaluation_metrics.png` in the project root.

> **Note:** `evaluate_model.py` was written for the local dataset-driven version of the model. To run it against the live MySQL-backed model, update the `RoomPlannerModel` instantiation to pass `DB_CONFIG` instead of a file path.

---

## Logging

The application logs at `INFO` level to two destinations simultaneously:

- **Console** — standard output
- **File** — `app.log` in the project root

Log format:

```
2026-03-04 12:00:00,000 - room-planner-api - INFO - Room Planner Model initialized successfully
```

---

## Error Responses

| HTTP Status | Meaning |
|---|---|
| `400 Bad Request` | One or more required fields are missing from the request body |
| `404 Not Found` | Product not found, or no related products exist for the given ID |
| `500 Internal Server Error` | Model failed to initialize, or an unexpected error occurred |

All error responses follow the format:

```json
{
  "error": "Description of the error"
}
```
