import cv2
import numpy as np

def detect_color_and_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    roi = hsv[cy-10:cy+10, cx-10:cx+10]

    avg_hsv = np.mean(roi.reshape(-1, 3), axis=0)
    h_val, s_val, v_val = avg_hsv

    # Print the HSV values in the console
    print(f"H: {h_val:.2f}, S: {s_val:.2f}, V: {v_val:.2f}")

    # Detect color
    color = 'UNKNOWN'
    color_bgr = (0, 255, 255)  # Default to yellow for unknown

    # Red (two ranges in HSV)
    if ((0 <= h_val <= 10) or (160 <= h_val <= 180)) and s_val > 70 and v_val > 50:
        color = 'RED'
        color_bgr = (0, 0, 255)
    # Yellow
    elif 20 <= h_val <= 35 and s_val > 70 and v_val > 50:
        color = 'YELLOW'
        color_bgr = (0, 255, 255)
    # Green
    elif 40 <= h_val <= 85 and s_val > 70 and v_val > 50:
        color = 'GREEN'
        color_bgr = (0, 255, 0)
    # Blue
    elif 100 <= h_val <= 130 and s_val > 70 and v_val > 50:
        color = 'BLUE'
        color_bgr = (255, 0, 0)
    # Black (low V, low S)
    elif v_val < 50 and s_val < 60:
        color = 'BLACK'
        color_bgr = (0, 0, 0)
    # White (high V, low S)
    elif v_val > 200 and s_val < 40:
        color = 'WHITE'
        color_bgr = (255, 255, 255)
    # Purple (magenta/violet)
    elif 130 <= h_val <= 160 and s_val > 60 and v_val > 50:
        color = 'PURPLE'
        color_bgr = (255, 0, 255)
    else:
        print("Erreur: couleur non détectée")

    return color, color_bgr, h_val, s_val, v_val

# Start the camera
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        color, color_bgr, h_val, s_val, v_val = detect_color_and_hsv(frame)

        # Draw the ROI
        cv2.rectangle(frame, (310, 230), (330, 250), (0, 0, 0), 4)

        # Show color name in the detected color
        cv2.putText(frame, f"Color: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_bgr, 2)

        # Show HSV values
        # cv2.putText(frame, f"H: {int(h_val)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        # cv2.putText(frame, f"S: {int(s_val)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        # cv2.putText(frame, f"V: {int(v_val)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Show the frame
        cv2.imshow("Live HSV Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
