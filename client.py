"""
CLI client for the HE-MNIST server.

Usage
-----
  python client.py                        # random 28×28 noise input
  python client.py --image path/to/img   # real image file
  python client.py --mode plain          # plaintext endpoint (faster, for dev)
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time

import numpy as np
import requests
import tenseal as ts
from PIL import Image

from he_utils import create_context

SERVER = "http://127.0.0.1:5000"


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return ((arr - 0.1307) / 0.3081).flatten()


def he_predict(flat: np.ndarray, context: ts.Context) -> tuple[int, float]:
    enc_x   = ts.ckks_vector(context, flat.tolist())
    payload = {"ciphertext": base64.b64encode(enc_x.serialize()).decode()}

    t0  = time.perf_counter()
    res = requests.post(f"{SERVER}/predict", json=payload, timeout=120)
    rtt = (time.perf_counter() - t0) * 1000

    res.raise_for_status()
    data    = res.json()
    enc_out = ts.ckks_vector_from(context, base64.b64decode(data["result"]))
    scores  = np.array(enc_out.decrypt()[:10])
    pred    = int(np.argmax(scores))
    server_ms = data.get("elapsed_ms", "n/a")
    print(f"Server HE time : {server_ms} ms")
    print(f"Round-trip time: {rtt:.0f} ms")
    return pred, float(scores[pred])


def plain_predict(img_path: str) -> int:
    with open(img_path, "rb") as f:
        res = requests.post(f"{SERVER}/predict_plain", files={"image": f}, timeout=30)
    res.raise_for_status()
    data = res.json()
    probs = data["probabilities"]
    print("Class probabilities:")
    for i, p in enumerate(probs):
        bar = "|" * int(p / 2)
        print(f"  {i}: {bar:<50} {p:5.1f}%")
    return data["prediction"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Path to a PNG/JPEG image (28*28 recommended)")
    ap.add_argument("--mode",  choices=["he", "plain"], default="he",
                    help="'he' = encrypted inference  |  'plain' = plaintext demo")
    args = ap.parse_args()

    if args.mode == "plain":
        if not args.image:
            print("--mode plain requires --image", file=sys.stderr)
            sys.exit(1)
        pred = plain_predict(args.image)
        print(f"\nPrediction: {pred}")
        return

    # HE mode
    if args.image:
        flat = load_image(args.image)
        print(f"Loaded image: {args.image}")
    else:
        flat = np.random.rand(784).astype(np.float32)
        print("Using random noise input (no --image provided)")

    context = create_context()
    print("Sending encrypted inference request …")
    pred, score = he_predict(flat, context)
    print(f"\nPrediction: {pred}  (raw logit: {score:.3f})")


if __name__ == "__main__":
    main()