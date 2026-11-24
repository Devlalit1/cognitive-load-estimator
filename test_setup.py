import cv2, mediapipe, numpy, sklearn, streamlit
print("Libraries OK")
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print("Webcam OK" if ok else "Webcam FAILED")
cap.release()
