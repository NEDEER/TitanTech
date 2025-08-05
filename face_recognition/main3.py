import cv2
import face_recognition
import os
import cvzone

# ! Load reference images and names
images_path = 'C:/Users/neder/Desktop/coinpapier/peoples'
images = []
dataset_encodings = []
names = []

for i in os.listdir(images_path):
    img = cv2.imread(os.path.join(images_path, i))
    images.append(img)
    name = i.split('.')[0]
    names.append(name)

# ! Encode reference images
for photo in images:
    rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        dataset_encodings.append(encodings[0])
    else:
        dataset_encodings.append(None)

# ! Load test images
test_path = 'C:/Users/neder/Desktop/coinpapier/test'
test_images = []

for i in os.listdir(test_path):
    test_img = cv2.imread(os.path.join(test_path, i))
    test_images.append(test_img)

# ! Process test images
for testpic in test_images:
    test_rgb = cv2.cvtColor(testpic, cv2.COLOR_BGR2RGB)
    test_face_locations = face_recognition.face_locations(test_rgb)
    test_face_encodings = face_recognition.face_encodings(test_rgb)

    for test_encoding, (y1, x2, y2, x1) in zip(test_face_encodings, test_face_locations):
        for counter, ref_encoding in enumerate(dataset_encodings):
            if ref_encoding is None:
                continue

            result = face_recognition.compare_faces([ref_encoding], test_encoding)[0]
            confidence = 1 - face_recognition.face_distance([ref_encoding], test_encoding)[0]

            if result:
                print(names[counter], confidence)
                cvzone.putTextRect(testpic, f"{names[counter]} {round(confidence, 2)}", (x1, y1 - 20), colorR=(0, 0, 255))
                w, h = x2 - x1, y2 - y1
                bbox = (x1, y1, w, h)
                cvzone.cornerRect(testpic, bbox, colorR=(0, 0, 255), colorC=(0, 0, 255), l=20, t=4)

    cv2.imshow('Detected Face', testpic)
    cv2.waitKey(0)

cv2.destroyAllWindows()