from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from model import RoomPlannerModel

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("room-planner-api")

app = Flask(__name__)
CORS(app)

# === UPDATE THESE WITH YOUR ACTUAL MYSQL CREDENTIALS ===
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'thilina_ecommerce',
    'port': 3306,
    'charset': 'utf8mb4',
    'autocommit': True
}

# Initialize model
try:
    model = RoomPlannerModel(DB_CONFIG)
    logger.info("Room Planner Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    model = None


@app.route('/api/search', methods=['POST'])
def search():
    if model is None:
        return jsonify({'error': 'Service unavailable'}), 500
    try:
        data = request.json
        room_size = data.get('roomSize')
        room_type = data.get('roomType')
        style = data.get('style')
        budget = data.get('budget')
        limit = int(data.get('limit', 5))

        if not all([room_size, room_type, style, budget]):
            return jsonify({'error': 'Missing required parameters'}), 400

        result = model.search(room_size, room_type, style, budget, limit)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/product/<product_id>', methods=['GET'])
def get_product(product_id):
    if model is None:
        return jsonify({'error': 'Service unavailable'}), 500
    try:
        product_data = model.get_product_by_id(product_id)
        if product_data is None:
            return jsonify({'error': 'Product not found'}), 404
        return jsonify(product_data)
    except Exception as e:
        logger.error(f"Product fetch error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/product/<product_id>/related', methods=['GET'])
def get_related_products(product_id):
    """
    New Endpoint: Returns up to 10 ML-based related products
    """
    if model is None:
        return jsonify({'error': 'Service unavailable'}), 500
    try:
        limit = int(request.args.get('limit', 10))
        related = model.get_related_products(product_id, limit=min(limit, 10))

        if not related:
            return jsonify({'error': 'Product not found or no related products'}), 404

        return jsonify({
            'product_id': product_id,
            'related_products': related
        })
    except Exception as e:
        logger.error(f"Related products error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/filters', methods=['GET'])
def get_filters():
    if model is None:
        return jsonify({'error': 'Service unavailable'}), 500
    try:
        return jsonify(model.get_available_filters())
    except Exception as e:
        logger.error(f"Filters error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    status = "healthy" if model is not None else "degraded"
    return jsonify({
        'status': status,
        'message': 'Room Planner API is running'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)