from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Load YOLO model
BURGER_PATH = "/static/clean_burger.png"  # Path to burger image

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO detection
    results = model(image_rgb)

    # Load burger image
    if not os.path.exists(BURGER_PATH):
        return jsonify({'error': 'Burger image not found'}), 500
    
    burger = Image.open(BURGER_PATH).convert("RGBA")
    image_pil = Image.fromarray(image_rgb)

    # Overlay burgers on detected hands/spoons
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            burger_resized = burger.resize((x2 - x1, y2 - y1))
            image_pil.paste(burger_resized, (x1, y1), burger_resized)

    # Save final image
    output_path = "output.png"
    image_pil.save(output_path)
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
