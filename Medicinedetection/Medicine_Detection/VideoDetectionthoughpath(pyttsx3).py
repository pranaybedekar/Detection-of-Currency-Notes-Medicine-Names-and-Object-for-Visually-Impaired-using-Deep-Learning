import cv2
import torch
import numpy as np
import pyttsx3

path='C:/Users/Admin/PycharmProjects/Medicinedetection/Medicine_Detection/yolov5/runs/train/exp/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)
engine = pyttsx3.init()

cap=cv2.VideoCapture('C:/Users/Admin/PycharmProjects/Medicinedetection/Medicine_Detection/yolov5/runs/train/exp/test_videos/download-33_xUg1hPpB.mp4')
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1280,720))
    results=model(frame)
    frame=np.squeeze(results.render())
    cv2.imshow("FRAME",frame)


    def speak(text):
        engine.say(text)
        engine.runAndWait()
        results = model(frame)


    for detection in results.xyxy[0]:
        class_name = model.names[int(detection[5])]
        speak(class_name)

    if cv2.waitKey(1)&0xFF==27:
        break
cap.realese()
cv2.destroyAllWindows()