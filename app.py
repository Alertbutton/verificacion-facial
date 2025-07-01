from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import requests

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "API de verificación facial activa"
    
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
        img1 = data.get("img1")
        img2 = data.get("img2")

        if not img1 or not img2:
            return jsonify({"error": "Faltan img1 o img2 en el cuerpo del request"}), 400

        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=True)

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
