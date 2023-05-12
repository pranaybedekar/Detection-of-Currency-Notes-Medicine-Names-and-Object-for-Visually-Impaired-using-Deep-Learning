import cv2
import torch
import time
#import detect

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/Admin/PycharmProjects/Currencydetection/Currency_Detection/yolov5/runs/train/exp/weights/best.pt')
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    results = model(frame)

    boxes = results.xyxy[0].tolist()
    labels = results.names[0]
    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)

        if labels and labels[int(box[5])] == 'currency':
            filename = time.strftime("%Y%m%d-%H%M%S") + '.jpg'
            cv2.imwrite(os.path.join('images', filename), frame)
        #img_name = 'object_{}.jpg'.format(time.strftime('%Y%m%d_%H%M%S'))
        #cv2.imwrite(img_name, frame)

    cv2.imshow('object Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()