from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
import os
from PIL import Image

app = Flask(__name__)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

BURGER_PATH = "burger.png"  # Path to burger image

def overlay_burger(image, hand_landmarks):
    print("Overlaying burger on detected hands...")
    burger = cv2.imread(BURGER_PATH, cv2.IMREAD_UNCHANGED)
    if burger is None:
        print("Error: Burger image not found!")
        return image
    
    burger = cv2.resize(burger, (100, 100))  # Resize burger
    for hand in hand_landmarks:
        x, y = int(hand.landmark[9].x * image.shape[1]), int(hand.landmark[9].y * image.shape[0])
        print(f"Hand detected at: ({x}, {y})")
        
        h, w, _ = burger.shape
        y1, y2 = y - h // 2, y + h // 2
        x1, x2 = x - w // 2, x + w // 2
        
        if 0 <= x1 < image.shape[1] and 0 <= x2 < image.shape[1] and 0 <= y1 < image.shape[0] and 0 <= y2 < image.shape[0]:
            print("Adding burger to hand position...")
            image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 0.5, burger, 0.5, 0)
    
    return image

@app.route('/')
def home():
    return jsonify({"message": "Flask is running!"})

@app.route('/routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(str(rule))
    return jsonify({"routes": routes})

@app.route('/upload', methods=['POST'])
def upload():
    print("Received image upload request...")
    if 'image' not in request.files:
        print("No image uploaded!")
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    print("Processing image...")
    image_np = np.array(Image.open(file).convert('RGB'))
    
    results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    if results.multi_hand_landmarks:
        print("Hands detected in image!")
        image_np = overlay_burger(image_np, results.multi_hand_landmarks)
    else:
        print("No hands detected!")
    
    output_path = "output.png"
    Image.fromarray(image_np).save(output_path)
    print("Image processing complete. Sending response...")
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    print(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
