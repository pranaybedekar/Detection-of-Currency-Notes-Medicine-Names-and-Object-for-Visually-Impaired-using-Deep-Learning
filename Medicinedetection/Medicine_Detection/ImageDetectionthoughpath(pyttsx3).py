
import pyttsx3
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Admin/PycharmProjects/Medicinedetection/Medicine_Detection/yolov5/runs/train/exp/weights/best.pt')

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

img = 'C:/Users/Admin/PycharmProjects/Medicinedetection/Medicine_Detection/yolov5/runs/train/exp/test_images1/Moncel FX_-1-_jpg.rf.63a044dd2f72e9884c281a8112d0f6ca (182).jpg'

results = model(img)

for detection in results.xyxy[0]:

    class_name = model.names[int(detection[5])]
    speak(class_name)


