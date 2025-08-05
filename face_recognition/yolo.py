from ultralytics import YOLO 
import cv2 
import cvzone
names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
    23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
    43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
    47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
    63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
    75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'}
model = YOLO('C:\\Users\\neder\\Documents\\yolo\\yolov8n.pt')

img = cv2.imread('C:/Users/neder/Desktop/smartGlasses/test1.jpg')
# img = cv2.resize(img , (0,0) , None ,0.5 , 0.5)
results = model.predict( img, save=False)

results = results[0]
print(results.boxes)
# print(results.boxes.names)

for box in results.boxes :
    x1,y1,x2,y2=box.xyxy[0]
    x1=int(x1)
    y1=int(y1)
    x2=int(x2)
    y2=int(y2)
    w=x2-x1
    h=y2-y1
    # print(box)
    # print(x1,y1,x2,y2)
    confidance=round(float(box.conf[0]),2)
    # print(confidance)
    id=int(box.cls[0])
    if (confidance>0.5):
        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),thickness=1,lineType=cv2.FONT_ITALIC)
        cvzone.cornerRect(img,(x1,y1,w,h))
        cvzone.putTextRect(img, f"{names[id]} {int(round(confidance * 100, 2))}%", (x1, y1 - 15), scale=2, thickness=2, offset=1)

cv2.imshow('image' , img)
cv2.waitKey(0)
# results[0].show()

