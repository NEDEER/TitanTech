import cv2  # OpenCV for image processing
import face_recognition  # Library for facial recognition
import os  # To interact with the operating system (file system)
import cvzone  # Library to enhance OpenCV visuals (rectangles, text, etc.)

# ! Reference images folder path
images_path = 'C:/Users/neder/Desktop/coinpapier/peoples'
images = []  # List to store all loaded images
dataset_encodings = []  # List to store face encodings from known images
names = []  # List to store names associated with each image

# Load images and extract names from file names
for i in os.listdir(images_path):
    img = cv2.imread(os.path.join(images_path, i))  # Use os.path.join for safety
    if img is None:
        print(f"Warning: Could not read image {i}. Skipping.")
        continue
    images.append(img)  # Store image
    name = i.split('.')[0]  # Extract name from filename (e.g., "jack_Ma.jpg" → "jack_Ma")
    names.append(name)

# Encode each image (only the first face found in each image)
for idx, photo in enumerate(images):
    rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (face_recognition uses RGB)
    encodings = face_recognition.face_encodings(rgb)
    if len(encodings) > 0:
        enc = encodings[0]  # Get the 128-dimension face encoding
        dataset_encodings.append(enc)  # Store encoding
    else:
        print(f"Warning: No face found in reference image '{names[idx]}'. Skipping this image.")
        # To keep names and encodings in sync, remove the name as well
        names.pop(idx)
        # Do not append to dataset_encodings

# ! Load test image for recognition
test_pic = cv2.imread('C:/Users/neder/Desktop/coinpapier/test/ndr.jpg')
if test_pic is None:
    raise FileNotFoundError("Test image not found at the specified path.")

test_rgb = cv2.cvtColor(test_pic, cv2.COLOR_BGR2RGB)  # Convert test image to RGB

# Get encoding(s) of face(s) in test image
test_face_locations = face_recognition.face_locations(test_rgb)
test_encodings = face_recognition.face_encodings(test_rgb, test_face_locations)

if len(test_encodings) == 0 or len(test_face_locations) == 0:
    print("No face found in the test image.")
    cv2.imshow('image', test_pic)
    cv2.waitKey(0)
else:
    # For each face found in the test image
    for test_encoding, (y1, x2, y2, x1) in zip(test_encodings, test_face_locations):
        found_match = False
        for counter, enc in enumerate(dataset_encodings):
            results = face_recognition.compare_faces([enc], test_encoding)[0]  # True if match found
            conf = 1 - face_recognition.face_distance([enc], test_encoding)[0]  # Confidence score (higher is better)

            if results:
                found_match = True
                print(names[counter], conf)  # Print matched name and confidence

                # Display name and confidence on the image
                cvzone.putTextRect(test_pic, names[counter] + ' ' + str(round(conf, 2)), (x1, y1 - 20), colorR=(0, 0, 255))

                # Draw a rectangle with corners around the detected face
                w = x2 - x1
                h = y2 - y1
                bbox = (x1, y1, w, h)
                cvzone.cornerRect(test_pic, bbox, colorR=(0, 0, 255), colorC=(0, 0, 255), l=20, t=4)

                # Load information text file based on matched name
                info_file = None
                if names[counter] == 'jack_Ma':
                    info_file = 'C:/Users/neder/Desktop/coinpapier/face_recognition/jack.txt'
                elif names[counter] == 'kais_saied':
                    info_file = 'C:/Users/neder/Desktop/coinpapier/face_recognition/kaies.txt'
                elif names[counter] == 'Elon_Musk':
                    info_file = 'C:/Users/neder/Desktop/coinpapier/face_recognition/Elon.txt'

                # Read and display the content of the matched person's file
                if info_file is not None and os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as file:
                        text = file.readlines()
                        for idx, line in enumerate(text, start=1):
                            cv2.putText(test_pic, line.strip(), (40, 40 * idx), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
                else:
                    print(f"Info file for {names[counter]} not found or not specified.")
        if not found_match:
            print("No match found for detected face.")

    # Display the final image with annotation
    cv2.imshow('image', test_pic)
    cv2.waitKey(0)  # Wait for a key press to close the image window
