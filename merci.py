import cv2
import os
from cvzone.FaceDetectionModule import FaceDetector
from datetime import datetime

# Create a folder to save screenshots if it doesn't exist
screenshot_folder = "screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

def save_screenshot(img):
    # Generate a unique filename based on the current time
    filename = os.path.join(screenshot_folder, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(filename, img)
    print(f"Screenshot saved: {filename}")

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, img = cap.read()
    if not success:
        continue
    # returns bounding boxes for the face 
    img, bBoxes = detector.findFaces(img)

    # Add the text overlay with blue color
    cv2.putText(img, 'MERCI POUR VOTRE ATTENTION', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if bBoxes:
        save_screenshot(img)  # Save screenshot when a face is detected

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
