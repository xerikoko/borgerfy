import logging
import os
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

# Configure logging for better debugging
log_file = "debug.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)  # Ensure Flask logs debug messages

BURGER_PATH = "static/burger.png"  # Ensure burger is placed inside /static/

# Check if burger image exists at startup
if not os.path.exists(BURGER_PATH):
    print(f"❌ ERROR: {BURGER_PATH} not found in the server!", flush=True)
    logging.error(f"Burger image not found at {BURGER_PATH}!")
else:
    print(f"✅ Burger image found at {BURGER_PATH}", flush=True)
    logging.debug(f"Burger image found at {BURGER_PATH}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=6,  # Allow more hands to be detected
    min_detection_confidence=0.1,  # Lower confidence threshold
    min_tracking_confidence=0.1  # Improve detection over multiple frames
)
mp_draw = mp.solutions.drawing_utils

def overlay_burger(image, hand_landmarks):
    print("🔍 Overlaying burger on detected hands...", flush=True)
    logging.debug("Overlaying burger on detected hands...")

    burger = cv2.imread(BURGER_PATH, cv2.IMREAD_UNCHANGED)
    if burger is None:
        print("❌ ERROR: Burger image not found!", flush=True)
        logging.error("Burger image not found!")
        return image

    burger = cv2.resize(burger, (150, 150))  # Increase burger size for visibility

    for i, hand in enumerate(hand_landmarks):
        x, y = int(hand.landmark[9].x * image.shape[1]), int(hand.landmark[9].y * image.shape[0])
        print(f"👉 Hand {i+1} detected at: ({x}, {y})", flush=True)
        logging.debug(f"Hand {i+1} detected at: ({x}, {y})")

        h, w, _ = burger.shape
        y1, y2 = y - h // 2, y + h // 2
        x1, x2 = x - w // 2, x + w // 2

        if 0 <= x1 < image.shape[1] and 0 <= x2 < image.shape[1] and 0 <= y1 < image.shape[0] and 0 <= y2 < image.shape[0]:
            print(f"✅ Adding burger to hand {i+1} at ({x1}, {y1}) - ({x2}, {y2})", flush=True)
            logging.debug(f"Adding burger to hand {i+1} at ({x1}, {y1}) - ({x2}, {y2})")

            if burger.shape[-1] == 4:  # Handle transparent burger image
                overlay = burger[:, :, :3]  # RGB
                mask = burger[:, :, 3] / 255.0  # Alpha mask

                for c in range(3):
                    image[y1:y2, x1:x2, c] = (1 - mask) * image[y1:y2, x1:x2, c] + mask * overlay[:, :, c]
            else:
                image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 0.5, burger, 0.5, 0)

    return image

@app.route('/upload', methods=['POST'])
def upload():
    print("🔍 Received image upload request...", flush=True)
    logging.debug("Received image upload request...")

    if 'image' not in request.files:
        print("❌ Error: No image found in request files!", flush=True)
        logging.error("No image uploaded!")
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    print(f"✅ Received file: {file.filename}", flush=True)
    logging.debug(f"Received file: {file.filename}")

    image_np = np.array(Image.open(file).convert('RGB'))
    print(f"🖼 Image size: {image_np.shape}", flush=True)
    logging.debug(f"Image size: {image_np.shape}")

    processed_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = hands.process(processed_image)

    if results.multi_hand_landmarks:
        print(f"✅ Hands detected! Number of hands: {len(results.multi_hand_landmarks)}", flush=True)
        logging.debug(f"Hands detected! Number of hands: {len(results.multi_hand_landmarks)}")

        for i, hand in enumerate(results.multi_hand_landmarks):
            x, y = int(hand.landmark[9].x * image_np.shape[1]), int(hand.landmark[9].y * image_np.shape[0])
            print(f"👉 Hand {i+1} detected at: ({x}, {y})", flush=True)
            logging.debug(f"Hand {i+1} detected at: ({x}, {y})")

        image_np = overlay_burger(image_np, results.multi_hand_landmarks)
    else:
        print("❌ No hands detected in the image!", flush=True)
        logging.warning("No hands detected in the image!")

    output_path = "output.png"
    Image.fromarray(image_np).save(output_path)
    print("✅ Image processing complete. Sending response...", flush=True)
    logging.debug("Image processing complete. Sending response...")

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    print(f"🚀 Starting Flask app on port {port}...", flush=True)
    logging.debug(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
