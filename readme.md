# NumPy Neural Network (From Scratch): MNIST Digit Recognizer

![Python](https://img.shields.io/badge/Python-3.12-blue)
![NumPy](https://img.shields.io/badge/NumPy-Pure%20Math-013243?logo=numpy)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)

A complete, end-to-end Multi-Layer Perceptron (MLP) built **entirely from scratch** using pure linear algebra and calculus in NumPy. No TensorFlow, no PyTorch, no black boxes. 

This project trains a 2-layer deep neural network on the MNIST dataset to recognize handwritten digits and includes a live OpenCV webcam pipeline for real-time, real-world inference.

## Features
* **Zero-Framework Math Engine:** Forward propagation, backpropagation (Chain Rule), and gradient descent implemented strictly using `numpy.dot()` and matrix operations.
* **Non-Linear Architecture:** 784 Input Nodes -> 128 Hidden Nodes (ReLU) -> 10 Output Nodes (Softmax).
* **Decoupled Architecture:** Trains the model once, saves the optimized parameters (`.npz`), and loads them instantly for lightweight inference.
* **Live Camera Inference:** Uses OpenCV to capture real-time webcam video, applies Gaussian blurring and Otsu's Thresholding, and feeds live frames into the custom neural network.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone 
   cd DIGIT_RECOG_LOGISTICREG

2. **Create a virtual environment:**
    ```bash

    python3 -m venv .venv
    source .venv/bin/activate

3. **Install the dependencies:**
    ```bash

    pip install numpy opencv-python scikit-learn matplotlib

    (Note: scikit-learn is strictly used to fetch the MNIST dataset efficiently, not for model training).

## How to Use

This project is broken into three distinct scripts:
1. **Train the Brain (model.py)**

    Run this first. It downloads the MNIST images, scales them, and trains the weights using Cross-Entropy Loss.
    ```bash

    python3 train.py

    Outputs: A my_mnist_model.npz file containing the optimized W and b matrices. Achieved 95.69% test accuracy(max).

2. **Static Image Checker (checker.py)**

    Test the network by using your live camera feed.
    (Note: If inbuilt webcam not present follow these steps).
    ```bash

    chmod +x ./camera/dependencies.sh
    ./camera/dependencies.sh
    chmod +x ./camera/camerarun.sh
    ./camera/camerarun.sh

    Once it is done we can move on and run

    ```bash

    python3 checker.py

    Outputs: Opens a real-time OpenCV window. Draw a digit on white paper with a thick black marker, hold it inside the targeting box, and watch the model predict the number. Press q to quit.

**The Math Behind the Code**

This network relies entirely on the following mathematical concepts:

    Forward Pass: Z1 = X · W1 + b1

        A1 = max(0, Z1) (ReLU Activation)

        Z2 = A1 · W2 + b2

        A2 = Softmax(Z2)

    Loss Function: Categorical Cross-Entropy.

    Backward Pass: Computes partial derivatives using the Chain Rule to find dW and db, propagating the error back through the ReLU derivative mask.

    Optimizer: Standard Gradient Descent.