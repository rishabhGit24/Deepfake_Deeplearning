import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained deepfake detection model
deepfake_detection_model = load_model('deepfake_detection_model.keras')

# Function to predict if an image frame contains a deepfake
def predict_deepfake(frame, model):
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    
    # Predict using the loaded model
    prediction = model.predict(processed_frame)
    
    # Get the class labels
    classes = ['Real', 'Fake']
    
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Get the predicted class label
    predicted_class = classes[predicted_class_index]
    
    return predicted_class

# Function to preprocess the frame before prediction
def preprocess_frame(frame):
    # Resize the frame to the desired dimensions
    resized_frame = cv2.resize(frame, (128, 128))
    
    # Convert the frame to RGB if it's in BGR format
    if resized_frame.shape[2] == 3:
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Normalize the pixel values to be between 0 and 1
    processed_frame = resized_frame / 255.0
    
    # Expand dimensions to match the model's expected input shape
    processed_frame = np.expand_dims(processed_frame, axis=0)
    
    return processed_frame

# Open a video capture stream (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Predict if the frame contains a deepfake
    if ret:
        prediction = predict_deepfake(frame, deepfake_detection_model)

        # Display the prediction on the frame
        cv2.putText(frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Real-Time Deepfake Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
