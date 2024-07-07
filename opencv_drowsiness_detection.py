import cv2
import pygame
import threading
import time

def play_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("music.wav")
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()

eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'
face_cascPath = 'haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

cap = cv2.VideoCapture(0)
alarm_thread = None
start_time = None
blink_durations = []

# Calibration Phase
calibration_duration = 15
calibration_complete = False
start_calibration_time = time.time()

def draw_text_with_background(img, text, position, font, scale, color, thickness, background_color, padding=5):
    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Define the box coordinates
    box_coords = ((position[0] - padding, position[1] + baseline - padding), 
                  (position[0] + text_width + padding, position[1] - text_height - padding))
    
    # Draw the background rectangle
    cv2.rectangle(img, box_coords[0], box_coords[1], background_color, cv2.FILLED)
    
    # Draw the text
    cv2.putText(img, text, position, font, scale, color, thickness)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (600, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_region = gray[y:y + h, x:x + w]
            eyes = eyeCascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

            if len(eyes) == 0:
                if start_time is None:
                    start_time = time.time()
                else:
                    elapsed_time = time.time() - start_time
            else:
                if start_time is not None:
                    blink_duration = time.time() - start_time
                    blink_durations.append(blink_duration)
                    start_time = None
                print("Eyes detected")

    # Display calibration message with background
    draw_text_with_background(img, "Calibration: Please blink naturally for 15 seconds.", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, (0, 0, 0))

    cv2.imshow('Calibration', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        calibration_complete = True
        break

    if time.time() - start_calibration_time > calibration_duration:
        calibration_complete = True
        break

cv2.destroyWindow('Calibration')

# Calculate average blink duration and set drowsiness threshold
if blink_durations:
    avg_blink_duration = sum(blink_durations) / len(blink_durations)
    DROWSINESS_THRESHOLD = avg_blink_duration * 2  # Customize the multiplier if needed
    print(f"Calibration complete. Average blink duration: {avg_blink_duration:.2f} seconds. Drowsiness threshold set to: {DROWSINESS_THRESHOLD:.2f} seconds.")
else:
    DROWSINESS_THRESHOLD = 1.5  # Default threshold if no blinks were detected during calibration
    print("Calibration failed. Using default drowsiness threshold.")

# Real-Time Monitoring Phase
start_time = None
while calibration_complete:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (600, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_region = gray[y:y + h, x:x + w]
            eyes = eyeCascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

            if len(eyes) == 0:
                if start_time is None:
                    start_time = time.time()
                else:
                    elapsed_time = time.time() - start_time
                    if elapsed_time > DROWSINESS_THRESHOLD:
                        if alarm_thread is None or not alarm_thread.is_alive():
                            alarm_thread = threading.Thread(target=play_alarm)
                            alarm_thread.start()
                        print("Drowsiness detected! Be alert!")
            else:
                start_time = None
                stop_alarm()
                print("Eyes detected")

    else:
        start_time = None
        stop_alarm()

    cv2.imshow('Face Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()
pygame.quit()