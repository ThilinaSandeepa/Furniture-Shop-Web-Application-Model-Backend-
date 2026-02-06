# Virtual Room Planner Backend

This is the backend service for the Virtual Room Planner application. It provides an API that recommends furniture based on room size, room type, preferred style, and budget range.

## Setup and Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

To start the Flask server, run:

```bash
python app.py
```

The server will start on http://localhost:5000

## API Endpoints

### Search for Furniture Recommendations

**Endpoint:** `/api/search`
**Method:** POST
**Request Body:**

```json
{
  "roomSize": "small",
  "roomType": "living-room",
  "style": "modern",
  "budget": "low"
}
```

**Response:**

```json
{
  "title": "Modern Living Room Design",
  "description": "Based on your small living room with modern style and low budget, here are our recommendations:",
  "suggestions": [
    {
      "name": "Furniture Item Name",
      "price": 299,
      "description": "Item description",
      "image": "image_url.jpg",
      "link": "/product/1"
    }
  ],
  "tips": [
    "Design tip 1",
    "Design tip 2",
    "Design tip 3"
  ]
}
```

### Health Check

**Endpoint:** `/api/health`
**Method:** GET
**Response:**

```json
{
  "status": "ok",
  "message": "Virtual Room Planner API is running"
}
```

## Dataset

The furniture recommendations are based on the dataset located at `Dataset/dataset.xlsx`. The dataset contains furniture items with the following attributes:

- id: Unique identifier for the item
- name: Name of the furniture item
- description: Description of the item
- price: Price of the item
- image: URL or path to the item's image
- room_size: Size of the room (Small, Medium, Large)
- room_type: Type of room (Living Room, Bedroom, Dining)
- style: Style of the furniture (Modern, Classic, Minimalist)
- budget_range: Budget range (Low, Medium, High)
