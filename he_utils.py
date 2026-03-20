import numpy as np
import tenseal as ts
import torch


def create_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = 2 ** 40
    return ctx


def load_weights(path="mnist.pth"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "net.0.weight" in state:
        w1_key, b1_key, w2_key, b2_key = "net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"
    else:
        w1_key, b1_key, w2_key, b2_key = "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"
    return (
        state[w1_key].numpy().astype(np.float32),
        state[b1_key].numpy().astype(np.float32),
        state[w2_key].numpy().astype(np.float32),
        state[b2_key].numpy().astype(np.float32),
    )


def he_linear(enc_vec, W, b):
    result = enc_vec.mm(W.T.tolist())
    return result + b.tolist()


def poly_activation(enc):
    # f(x) = 0.125*x^2 + 0.5*x  computed manually to avoid polyval depth issues
    sq = enc * enc        # uses 1 multiplication level
    return sq * 0.125 + enc * 0.5


class HEInferenceEngine:
    def __init__(self, weight_path="mnist.pth"):
        self.W1, self.b1, self.W2, self.b2 = load_weights(weight_path)

    def predict(self, enc_x):
        enc = he_linear(enc_x, self.W1, self.b1)
        enc = poly_activation(enc)
        enc = he_linear(enc, self.W2, self.b2)
        return enc