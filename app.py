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
    burger = cv2.imread(BURGER_PATH, cv2.IMREAD_UNCHANGED)
    if burger is None:
        return image
    
    burger = cv2.resize(burger, (100, 100))  # Resize burger
    for hand in hand_landmarks:
        x, y = int(hand.landmark[9].x * image.shape[1]), int(hand.landmark[9].y * image.shape[0])
        
        h, w, _ = burger.shape
        y1, y2 = y - h // 2, y + h // 2
        x1, x2 = x - w // 2, x + w // 2
        
        if 0 <= x1 < image.shape[1] and 0 <= x2 < image.shape[1] and 0 <= y1 < image.shape[0] and 0 <= y2 < image.shape[0]:
            image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 0.5, burger, 0.5, 0)
    
    return image

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    image_np = np.array(Image.open(file).convert('RGB'))
    
    results = hands.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    if results.multi_hand_landmarks:
        image_np = overlay_burger(image_np, results.multi_hand_landmarks)
    
    output_path = "output.png"
    Image.fromarray(image_np).save(output_path)
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
