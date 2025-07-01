from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import requests

app = Flask(__name__)

def read_image_from_url(url):
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    img1_url = data.get('img1')
    img2_url = data.get('img2')

    if not img1_url or not img2_url:
        return jsonify({"error": "Faltan URLs de imagen"}), 400

    try:
        img1 = read_image_from_url(img1_url)
        img2 = read_image_from_url(img2_url)

        result = DeepFace.verify(img1, img2, enforce_detection=False)

        return jsonify({
            "verified": result["verified"],
            "distance": result["distance"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "API de verificación facial activa."

if __name__ == '__main__':
    app.run()
