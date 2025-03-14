import os
import cv2
import face_recognition
import pickle
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta

# MongoDB connection
client = MongoClient('mongodb+srv://jithindandyala:jithin2030@cluster0.ywkjq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['FaceRecognitionDB']
collection = db['FaceLogs']

# Load face encodings
with open("EncodeFile.p", 'rb') as file:
    encode_with_id = pickle.load(file)

encode_extract, stud_id = encode_with_id

# Load UI images
nosuer = cv2.imread('Resources/nouser.jpg')

# Check if the face was logged in the last 24 hours
def has_logged_recently(student_id):
    last_log = collection.find_one({"student_id": student_id}, sort=[("timestamp", -1)])
    if last_log:
        last_log_time = last_log["timestamp"]
        if datetime.now() - last_log_time < timedelta(hours=24):
            return True
    return False

# Log the face recognition
def log_face_recognition(student_id):
    if not has_logged_recently(student_id):
        collection.insert_one({
            "student_id": student_id,
            "timestamp": datetime.now()
        })

# Process frame for face recognition
def process_frame(frame):
    img_s = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    face_cur_frame = face_recognition.face_locations(img_s)
    encode_live = face_recognition.face_encodings(img_s, face_cur_frame)

    if not encode_live:
        print("⚠️ No face detected in frame.")
        return None

    for enc_face, face_loc in zip(encode_live, face_cur_frame):
        matches = face_recognition.compare_faces(encode_extract, enc_face)
        face_dis = face_recognition.face_distance(encode_extract, enc_face)

        if len(face_dis) == 0:
            print("⚠️ No matching faces in database.")
            return None

        match_ind = np.argmin(face_dis)
        threshold = 0.5

        if face_dis[match_ind] < threshold and matches[match_ind]:
            student_id = stud_id[match_ind]
            log_face_recognition(student_id)
            return student_id

    print("🔴 Unknown face detected.")
    return None

# Video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    student_id = process_frame(img)

    if student_id:
        print(f"🟢 Face identified! You are {student_id}")

    cv2.imshow("Camera Accessed", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Camera Accessed", cv2.WND_PROP_VISIBLE) < 1:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
