# 🔐 Homomorphic Encryption MNIST Digit Classifier

A privacy-preserving neural network for handwritten digit classification using Homomorphic Encryption (HE). It enables inference directly on encrypted data — **no access to raw inputs required**.

---

## 🚀 Overview

This project demonstrates secure ML by running an MNIST classifier on encrypted images. The model is trained in plaintext but evaluated in the encrypted domain, ensuring user data remains private.

---

## 🧠 Features

* 🔒 End-to-end encrypted inference
* 🧮 HE-friendly architecture (polynomial activations)
* ⚡ Optimized linear layers (diagonal method)
* 🧪 Evaluated on MNIST dataset

---

## 🏗 Architecture

```
Input (784) → Linear → Square → Linear → Output (10)
```

Designed to work within HE constraints:

* ReLU replaced with square activation
* Limited depth to control noise
* Linear layers adapted for encrypted computation

## ⚠️ Limitations & Future Work

Slower than standard inference

Limited model complexity

Future: better packing, deeper models, deployment as API
