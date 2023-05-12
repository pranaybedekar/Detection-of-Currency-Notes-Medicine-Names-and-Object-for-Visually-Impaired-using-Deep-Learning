import urllib.request
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import tensorflow_hub as hub


# Load the MobileNet model from TensorFlow Hub
model_url = "C:/Users/Admin/Desktop/performance tuning/detect.tflite"
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(320, 320, 3)),
    hub.KerasLayer(model_url, trainable=False)
])

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Define your class names
class_names = ['10', '20', '50', '100', '200', '500', '2000', 'Fifty', 'One Hundred', 'Ten', 'Twenty']

# Define the colors for the bounding boxes
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

# Download the image from the URL and decode it
url = "C:/Users/Admin/PycharmProjects/Currencydetection/Currency_Detection/yolov5/runs/train/exp/test_images1/download (7).jpg"
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
frame = cv2.imdecode(arr, -1)

# Resize the image to the input size expected by the model (e.g., 224x224)
resized_frame = cv2.resize(frame, (320, 320))

# Preprocess the image by converting it to a NumPy array and scaling its values
preprocessed_frame = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 127.5 - 1.0

# Get the predicted class probabilities from the model
predicted_probabilities = model.predict(preprocessed_frame)

# Get the index of the predicted class with the highest probability
predicted_class_index = np.argmax(predicted_probabilities)

# Get the name of the predicted class
predicted_class_name = class_names[predicted_class_index]

# Get the confidence score of the predicted class
confidence_score = predicted_probabilities[0][predicted_class_index]

# Speak the name and confidence score of the predicted class using the text-to-speech engine
engine.say(predicted_class_name + ' with confidence score ' + str(confidence_score))
engine.runAndWait()

# Display the image with the predicted class name overlaid on it
cv2.putText(frame, predicted_class_name + ' ' + str(confidence_score), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('frame', frame)

# Wait for a key event and then destroy the window
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
import cv2
import numpy as np
import pyttsx3
import tensorflow as tf

# Initialize the TFLite interpreter with your own TFLite model file
interpreter = tf.lite.Interpreter(model_path="C:/Users/Admin/Desktop/performance tuning/detect.tflite")
interpreter.allocate_tensors()

# Get the input and output tensors of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define your class names
class_names = ['10', '20', '50', '100', '200', '500', '2000', 'Fifty', 'One Hundred', 'Ten', 'Twenty']

# Define the colors for the bounding boxes
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to the input size expected by the model (e.g., 224x224)
    resized_frame = cv2.resize(frame, (320, 320))

    # Preprocess the frame by converting it to a NumPy array and scaling its values
    preprocessed_frame = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 127.5 - 1.0

    # Set the input tensor of the model to the preprocessed frame
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

    # Run inference on the input tensor
    interpreter.invoke()

    # Get the output tensor of the model
    output_tensor = interpreter.get_tensor(output_details[0]['index'])

    # Get the index of the predicted class with the highest probability
    predicted_class_index = np.argmax(output_tensor)

    # Get the name of the predicted class
    predicted_class_name = class_names[predicted_class_index]

    # Get the confidence score of the predicted class
    confidence_score = output_tensor[0][predicted_class_index]

    # Get the bounding box of the object detected by the model
    ymin, xmin, ymax, xmax = [int(i * 320) for i in output_tensor[0][5:9]]

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colors[predicted_class_index], 2)

    # Display the class name and confidence score above the bounding box
    label = predicted_class_name + ' ' + str(round(confidence_score * 100, 2)) + '%'
    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[predicted_class_index], 2)

    # Display the frame with the bounding box and class name overlaid on it
    cv2.imshow('frame', frame)

    # Speak the name and confidence score of the predicted class using the text-to-speech engine
    engine.say(predicted_class_name + ' with confidence score ' + str(round(confidence_score * 100, 2)) + '%')
    engine.runAndWait()

    # Break the loop if the user presses the ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
"""



"""
import cv2
import numpy as np
import pyttsx3
import tensorflow as tf

# Initialize the TFLite interpreter with your own TFLite model file
interpreter = tf.lite.Interpreter(model_path="C:/Users/Admin/Desktop/performance tuning/detect.tflite")
interpreter.allocate_tensors()

# Get the input and output tensors of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to the input size expected by the model (e.g., 224x224)
    resized_frame = cv2.resize(frame, (320, 320))

    # Preprocess the frame by converting it to a NumPy array and scaling its values
    preprocessed_frame = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 127.5 - 1.0

    # Set the input tensor of the model to the preprocessed frame
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

    # Run inference on the input tensor
    interpreter.invoke()

    # Get the output tensor of the model
    output_tensor = interpreter.get_tensor(output_details[0]['index'])

    # Get the index of the predicted class with the highest probability
    predicted_class_index = np.argmax(output_tensor)

    # Get the name of the predicted class
    class_names = ["10", "20", "50", "100", "200", "500", "2000" , "Fifty", "One Hundred", "Ten", "Twenty"]
    predicted_class_name = class_names[predicted_class_index]


    #predicted_class_name = "your_class_names_list[predicted_class_index]"

    # Speak the name of the predicted class using the text-to-speech engine
    engine.say(predicted_class_name)
    engine.runAndWait()

    # Display the frame with the predicted class name overlaid on it
    cv2.putText(frame, predicted_class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    # Break the loop if the user presses the ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()  """

