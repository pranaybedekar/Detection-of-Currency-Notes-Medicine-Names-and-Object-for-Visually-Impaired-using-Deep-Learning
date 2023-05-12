from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0) # for webcam
#cap.set(3, 1280)
#cap.set(4, 720)

cap = cv2.VideoCapture("C:/Users/Admin/PycharmProjects/Currencydetection/Currency_Detection/yolov5/runs/train/exp/test_videos1/production ID_4112668.mp4")   # for video

model = YOLO("C:/Users/Admin/PycharmProjects/Currencydetection/Currency_Detection/yolov5/runs/train/exp/weights/best.pt")
classNames = ['10', '20', '50', '100', '200', '500', '2000', 'Fifty', 'One Hundred', 'Ten', 'Twenty']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # for opencv
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            # Confidence (percentage)
            conf = math.ceil((box.conf[0]*100))/100

            # Class Name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)))

    cv2.imshow("Image",img)
    cv2.waitKey(1)