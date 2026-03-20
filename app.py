import base64
import io
import logging
import os
import time

import numpy as np
import tenseal as ts
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from he_utils import HEInferenceEngine, create_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__)
CORS(app)

WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "mnist.pth")
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError(f"Weights not found: {WEIGHT_PATH} — run train.py first")

log.info("Initialising HE context…")
_context = create_context()
log.info("Loading weights…")
_engine = HEInferenceEngine(WEIGHT_PATH)
log.info("Server ready.")


@app.route("/")
def index():
    html_path = os.path.join(STATIC_DIR, "index.html")
    log.info("Serving: %s (exists=%s)", html_path, os.path.exists(html_path))
    with open(html_path, encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html"}


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.perf_counter()
    try:
        body = request.get_json(silent=True) or {}
        if "ciphertext" not in body:
            return jsonify({"error": "missing 'ciphertext'"}), 400
        enc_x = ts.ckks_vector_from(_context, base64.b64decode(body["ciphertext"]))
        enc_out = _engine.predict(enc_x)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info("HE inference: %.0f ms", elapsed_ms)
        return jsonify({"result": base64.b64encode(enc_out.serialize()).decode(), "elapsed_ms": round(elapsed_ms, 1)})
    except Exception as exc:
        log.exception("predict failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict_plain", methods=["POST"])
def predict_plain():
    if "image" not in request.files:
        return jsonify({"error": "field 'image' required"}), 400
    try:
        img = Image.open(io.BytesIO(request.files["image"].read())).convert("L")
        arr = np.array(img.resize((28, 28), Image.LANCZOS), dtype=np.float32) / 255.0
        flat = ((arr - 0.1307) / 0.3081).flatten().tolist()
    except Exception as exc:
        return jsonify({"error": f"image error: {exc}"}), 400

    enc_out = _engine.predict(ts.ckks_vector(_context, flat))
    scores = np.array(enc_out.decrypt()[:10])
    pred = int(np.argmax(scores))
    e = np.exp(scores - scores.max())
    probs = (e / e.sum() * 100).round(1).tolist()
    return jsonify({"prediction": pred, "probabilities": probs})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG", "0") == "1")