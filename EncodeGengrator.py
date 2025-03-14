import cv2
import face_recognition
import pickle
import os
from pymongo import MongoClient
# MongoDB Initialization
client = MongoClient('mongourl')  # For local MongoDB
# client = MongoClient('your_mongo_atlas_connection_string')  # For MongoDB Atlas
db_mongo = client['FaceRecognitionDB']
collection = db_mongo['FaceEncodings']

# Load Images
folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
studentIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])

print(f"Number of images: {len(imgList)}")
print(f"Student IDs: {studentIds}")

# Function to Encode Faces
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if face was detected
            encodeList.append(encodings[0])
        else:
            print("No face detected in one of the images.")
            encodeList.append(None)  # Handle the case when no face is found
    return encodeList

# Generate Encodings
print("Encoding Started...")
encodeListKnown = findEncodings(imgList)

# Filter out any None values
filteredEncodings = [encode for encode in encodeListKnown if encode is not None]
filteredStudentIds = [studentIds[i] for i in range(len(encodeListKnown)) if encodeListKnown[i] is not None]

# Save Encodings to MongoDB
mongo_data = []
for i, encoding in enumerate(filteredEncodings):
    mongo_data.append({
        "student_id": filteredStudentIds[i],
        "encoding": encoding.tolist()
    })

collection.insert_many(mongo_data)

# Save Locally with Pickle
with open("EncodeFile.p", 'wb') as file:
    pickle.dump([filteredEncodings, filteredStudentIds], file)

print("Encoding Complete and Saved to MongoDB")
