import cv2
import numpy as np

def softmax(Z):
    exp_z = np.exp(Z-np.max(Z,axis=1,keepdims=True))
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)

def ReLU(Z):
    return np.maximum(0,Z)

model = np.load("finaldataset.npz")
W1 = model['W1']
W2 = model['W2']
b1 = model['b1']
b2 = model['b2']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Is your camera connected?")
        break

    # Flip the frame horizontally like a mirror (feels more natural)
    #frame = cv2.flip(frame, 1)

    # --- D. DEFINE THE SCANNING ZONE (Region of Interest) ---
    # We don't want the network looking at your face or the background. 
    # We draw a 300x300 box in the center of the screen and only feed that box to the model.
    height, width, _ = frame.shape
    top_left_y = int(height/2 - 150)
    top_left_x = int(width/2 - 150)
    bottom_right_y = int(height/2 + 150)
    bottom_right_x = int(width/2 + 150)

    # Draw a green rectangle on the screen to guide the user
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    # Crop the image to just the inside of the green box
    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # --- E. IMAGE PREPROCESSING ---
    # 1. Convert to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Add slight blur to smooth out camera noise
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    
    # 3. OTSU Thresholding: Automatically finds the best lighting threshold, 
    # and INVERTS it so the dark ink becomes white and the paper becomes pitch black.
    _, roi_thresh = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Resize to 28x28
    roi_resized = cv2.resize(roi_thresh, (28, 28))

    # 5. Scale and flatten for the neural network
    X_live = (roi_resized / 255.0).reshape(1, 784)

    Z1 = np.dot(X_live, W1) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    prediction = np.argmax(A2, axis=1)[0]
    confidence = np.max(A2) * 100

    # Only show the prediction if the model is somewhat confident (avoids flickering random numbers)
    if confidence > 60:
        text = f"Guess: {prediction} ({confidence:.1f}%)"
        cv2.putText(frame, text, (top_left_x, top_left_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("CAMERA FEED", frame)
    
    cv2.imshow("OUTPUT", cv2.resize(roi_thresh, (200, 200))) 

    # Press 'q' on your keyboard to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
