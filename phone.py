import cv2
import time
from datetime import datetime
import os
from ultralytics import YOLO
import torch


model = YOLO(r"yolov8s.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

"""
def process_mobile_detection(frame):
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            mobile_detected = False

            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]

            print(f"Detected: {label}, Confidence: {conf:.2f}")

            if conf < 0.8 or label == "phone":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            display_label = f"{label} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            mobile_detected = True
    
    return frame, mobile_detected
"""
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
def process_mobile_detection(frame):
    results = model(frame, verbose=False)
    mobile_detected = False 

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]

            print(f"Detected: {label}, Confidence: {conf:.2f}")

            if conf < 0.8:
                continue

            if label == "cell phone":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                display_label = f"{label} ({conf:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"log/phone_detected_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[ALERTE] Capture enregistrée dans {filename}")

                mobile_detected = True
            elif label == "person":
                mobile_detected = False 

    return frame, mobile_detected

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

print("Webcam ouverte avec succès. Démarrage de la détection de mobile...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de l'image.")
        break

    frame, mobile_detected = process_mobile_detection(frame)
    cv2.putText(frame, f"Mobile Detected: {mobile_detected}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Webcam with Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Fin de la détection. Fermeture de la webcam.")
        break



cap.release()  
cv2.destroyAllWindows()  


