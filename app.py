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

@app.route("/verify", methods=["POST"])
def verify():
    try:
        data = request.get_json()
        print("📥 Datos recibidos:", data)

        img1_url = data.get("img1")
        img2_url = data.get("img2")

        if not img1_url or not img2_url:
            print("❌ Faltan URLs")
            return jsonify({"error": "Faltan URLs de imagen"}), 400

        img1 = read_image_from_url(img1_url)
        img2 = read_image_from_url(img2_url)

        print("✅ Imágenes descargadas")

        result = DeepFace.verify(img1_path=img1, img2_path=img2, enforce_detection=True)

        print("✅ Resultado:", result)

        return jsonify({
            "verified": result["verified"],
            "distance": result["distance"]
        })

    except Exception as e:
        print("💥 ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    return "API de verificación facial activa."

if __name__ == '__main__':
    app.run()
