
import pyttsx3
import torch  #import torch library

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Admin/PycharmProjects/Currencydetection/Currency_Detection/yolov5/runs/train/exp/weights/best.pt')  #pretrained model for object detection  # ultralytics repo on github #custom dataset

engine = pyttsx3.init()  #initialize text to speech engine

def speak(text):  #define speak function convert and play the text as speech
    engine.say(text)
    engine.runAndWait()

img = 'C:/Users/Admin/PycharmProjects/Currencydetection/Currency_Detection/yolov5/runs/train/exp/test_images1/indcurrency.jpg'

results = model(img)

for detection in results.xyxy[0]:  #loop over each detection

    class_name = model.names[int(detection[5])] #get the class name of detected object
    speak(class_name) #speak function to convert and play the detected class name as speech


