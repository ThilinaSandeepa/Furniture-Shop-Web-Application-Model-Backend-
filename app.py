from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from threading import Lock

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

model = None
model_init_error = None
model_lock = Lock()

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

def get_model():
    global model, model_init_error

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        try:
            from model import RoomPlannerModel

            model = RoomPlannerModel(DB_CONFIG)
            model_init_error = None
            logger.info("Room Planner Model initialized successfully")
            return model
        except Exception as e:
            model_init_error = str(e)
            logger.error(f"Failed to initialize model: {model_init_error}")
            return None


@app.route('/api/search', methods=['POST'])
def search():
    current_model = get_model()
    if current_model is None:
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

        result = current_model.search(room_size, room_type, style, budget, limit)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/product/<product_id>', methods=['GET'])
def get_product(product_id):
    current_model = get_model()
    if current_model is None:
        return jsonify({'error': 'Service unavailable'}), 500
    try:
        product_data = current_model.get_product_by_id(product_id)
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
    current_model = get_model()
    if current_model is None:
        return jsonify({'error': 'Service unavailable'}), 500
    try:
        limit = int(request.args.get('limit', 10))
        related = current_model.get_related_products(product_id, limit=min(limit, 10))

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
    current_model = get_model()
    if current_model is None:
        return jsonify({'error': 'Service unavailable'}), 500
    try:
        return jsonify(current_model.get_available_filters())
    except Exception as e:
        logger.error(f"Filters error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    current_model = get_model()
    status = "healthy" if current_model is not None else "degraded"
    return jsonify({
        'status': status,
        'message': 'Room Planner API is running'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)