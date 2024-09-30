import os
import cv2
from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile

# Инициализация приложения FastAPI
app = FastAPI()

# Загрузка каскадов Хаара для лиц и глаз
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Функция для поиска лиц и глаз
def detect_largest_face_and_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    largest_face = None
    max_area = 0
    eyes_detected = False

    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            face_region = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_region)
            if len(eyes) >= 2:  # Оба глаза найдены
                largest_face = (x, y, w, h)
                max_area = area
                eyes_detected = True

    return largest_face, max_area, eyes_detected

# Функция сравнения лиц по площади
def compare_faces(file1_path, file2_path):
    face1, area1, eyes1 = detect_largest_face_and_eyes(file1_path)
    face2, area2, eyes2 = detect_largest_face_and_eyes(file2_path)

    if not (face1 and eyes1) or not (face2 and eyes2):
        return 0, "Error detecting faces or eyes in one or both images."

    difference = abs(area1 - area2) / max(area1, area2) * 100
    return difference, None

# Маршрут для сравнения лиц
@app.post("/compare-faces/")
async def compare_faces_endpoint(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp1, NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp2:
            # Сохранение временных файлов для обработки
            tmp1.write(await file1.read())
            tmp2.write(await file2.read())
            tmp1_path, tmp2_path = tmp1.name, tmp2.name

        # Сравнение лиц
        difference, error = compare_faces(tmp1_path, tmp2_path)

        # Удаление временных файлов
        os.remove(tmp1_path)
        os.remove(tmp2_path)

        if error:
            return {"difference": 0, "error": error}

        return {"difference": difference, "status": "passed" if difference > 15 else "failed"}

    except Exception as e:
        return {"difference": 0, "error": str(e)}
