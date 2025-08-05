from ultralytics import YOLO 
import cv2 
import cvzone
import speech_recognition
import pyttsx3 
import random 
import time
import numpy as np
speaker = pyttsx3.init()
speaker.setProperty('rate', 120)
speaker.setProperty('volume', 1)
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[0].id)
recognizer = speech_recognition.Recognizer()
Hello_dataset = ['hello', 'good morning', 'good evening', 'hi']
Bye_dataset = ['goodbye', 'take care', 'bye']
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

# Use webcam for live detection, or use an image for testing
cap = cv2.VideoCapture(0)  # Use 0 for default camera

active = False  # System starts inactive, waits for hello
last_spoken = {}  # To avoid repeating the same detection too fast

def get_position(x1, x2, img_width):
    center = (x1 + x2) // 2
    if center < img_width // 3:
        return "left"
    elif center > 2 * img_width // 3:
        return "right"
    else:
        return "center"

def speak(text, voice_id=0):
    speaker.setProperty('voice', voices[voice_id].id)
    speaker.say(text)
    speaker.runAndWait()

def listen_for_command():
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        try:
            audio = recognizer.listen(mic, timeout=3, phrase_time_limit=3)
            command = recognizer.recognize_google(audio).lower()
            return command
        except Exception:
            return None

while True:
    # Always listen for bye/hello commands
    command = listen_for_command()
    if command:
        if any(bye in command for bye in Bye_dataset):
            if active:
                speak(random.choice(Bye_dataset), voice_id=1)
                active = False
            continue
        elif any(hello in command for hello in Hello_dataset):
            if not active:
                speak(random.choice(Hello_dataset), voice_id=0)
                active = True
            continue

    if not active:
        # Show a waiting screen or just continue
        img = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Say 'hello' to start", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    ret, img = cap.read()
    if not ret:
        continue

    results = model.predict(img, save=False)[0]
    img_height, img_width = img.shape[:2]

    detected_this_frame = set()
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        confidence = round(float(box.conf[0]), 2)
        id = int(box.cls[0])
        name = names.get(id, "object")
        if confidence > 0.5:
            cvzone.cornerRect(img, (x1, y1, w, h))
            cvzone.putTextRect(img, f"{name} {int(round(confidence * 100, 2))}%", (x1, y1 - 15), scale=2, thickness=2, offset=1)
            position = get_position(x1, x2, img_width)
            key = (name, position)
            detected_this_frame.add(key)
            # Speak only if not spoken recently
            now = time.time()
            if key not in last_spoken or now - last_spoken[key] > 2:
                speak(f"{name} {position}", voice_id=0)
                last_spoken[key] = now

    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
