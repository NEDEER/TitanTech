from ultralytics import YOLO 
import cv2 
import cvzone
names = {0: '100', 1: '50', 2: '500'}

model = YOLO('C:\\Users\\neder\\Desktop\\coinpapier\\coin.pt')

img = cv2.imread('C:\\Users\\neder\\Desktop\\coinpapier\\testt.jpg')
#img = cv2.resize(img , (0,0) , None ,0.7 , 0.7)
results = model.predict( img, save=False)
print(results)
results = results[0]

for box in results.boxes :
    x1 , y1 , x2, y2 = box.xyxy[0]
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    w = x2 - x1 
    h = y2 - y1
    print(x1 , y1 , x2 , y2)
    confidence = round(float(box.conf[0]) , 2 )
    id = int(box.cls[0])
    # print(names[id] , confidence)
    # cv2.rectangle(img , (x1,y1) , (x2,y2) , (0,0,255) , 3
    cvzone.putTextRect(img , names[id]+' '+str(int(confidence*100))+' '+'%' , (x1,y1-15) , scale=2 , thickness=2 , offset=1)
    cvzone.cornerRect(img , (x1,y1,w,h))
    print(names[id])
def sum_coins(results, names):
    """
    Sums the total value of detected coins based on their class IDs.
    Args:
        results: The results object from YOLO model prediction (results[0]).
        names: Dictionary mapping class IDs to coin values as strings.
    Returns:
        total_sum: The sum of all detected coins as an integer.
        coin_counts: Dictionary with counts of each coin type.
    """
    # Convert string values in names to integers
    value_map = {k: int(v) for k, v in names.items()}
    total_sum = 0
    coin_counts = {v: 0 for v in value_map.values()}
    for box in results.boxes:
        id = int(box.cls[0])
        value = value_map.get(id, 0)
        total_sum += value
        coin_counts[value] += 1
    return total_sum, coin_counts

# Example usage after detection:
total, counts = sum_coins(results, names)
print(f"Total sum of coins: {total}")
print("Coin counts:", counts)
cv2.putText(img, f"Total sum of coins: {total}", (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
cv2.putText(img, f"coin counts: {counts}", (30, 300), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
cv2.imshow('image' , img)
cv2.waitKey(0)
# results[0].show()

