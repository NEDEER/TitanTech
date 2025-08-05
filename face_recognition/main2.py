import cv2 
import face_recognition
import os 
import cvzone 
# for test
cap = cv2.VideoCapture(0)
# !photo of reference 
images_path = 'C:/Users/neder/Desktop/FormationCV/Session 5/peoples'
images = []
dataset_encodings = []
names = []
for i in os.listdir(images_path):
    img = cv2.imread(images_path+'/'+i)
    images.append(img)
    name = i.split('.')[0]
    names.append(name)
# print(names)
for photo in images : 
    rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if len(encodings) > 0:
        enc = encodings[0]
        dataset_encodings.append(enc)
    else:
        print("Warning: No face found in one of the reference images. Skipping this image.")
# print(len(dataset_encodings))
# !test photo
while True:
    ret, frame = cap.read()

    if not ret:
        continue  # Skip if frame not read properly

    test_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(test_rgb)
    face_encodings = face_recognition.face_encodings(test_rgb, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        y1, x2, y2, x1 = face_location

        for counter, enc in enumerate(dataset_encodings):
            results = face_recognition.compare_faces([enc], face_encoding)[0]
            conf = 1 - face_recognition.face_distance([enc], face_encoding)[0]

            if results:
                print(names[counter], conf)
                cvzone.putTextRect(frame, names[counter] + ' ' + str(round(conf, 2)), (x1, y1 - 20), colorR=(0, 0, 255))
                w = x2 - x1
                h = y2 - y1
                bbox = (x1, y1, w, h)
                cvzone.cornerRect(frame, bbox, colorR=(0, 0, 255), colorC=(0, 0, 255), l=20, t=4)

                if names[counter] == 'jack_Ma':
                    file = open('C:/Users/neder/Desktop/FormationCV/Session 5/face_recognition/jack.txt')
                elif names[counter] == 'kais_saied':
                    file = open('C:/Users/neder/Desktop/FormationCV/Session 5/face_recognition/kaies.txt')
                elif names[counter] == 'Elon_Musk':
                    file = open('C:/Users/neder/Desktop/FormationCV/Session 5/face_recognition/Elon.txt')
                else:
                    file = None

                if file:
                    text = file.readlines()
                    for idx, line in enumerate(text):
                        cv2.putText(frame, line.strip(), (40, 40 * (idx + 1)), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                    file.close()

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Quit with 'q' key




