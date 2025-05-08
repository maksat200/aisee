import os
import uuid
import shutil
import json
import cv2
import datetime
import re
import easyocr

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from src.logic.customer_counter2 import AsyncPersonCounter

router = APIRouter()

VIDEO_DIR = 'src/assets/video'
RESULTS_DIR = 'src/assets/results'

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_timestamp_easyocr(frame):
    timestamp_region = frame[0:40, 0:300]
    reader = easyocr.Reader(['en'], gpu=False)  # можно включить GPU, если есть
    result = reader.readtext(timestamp_region)

    if result:
        raw_text = result[0][1]
        print("Raw OCR output:", raw_text)

        match = re.search(r'(\d{2})[-/](\d{2})[-/](\d{4})', raw_text)
        if match:
            month, day, year = match.groups()
            try:
                date_obj = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                weekday = date_obj.strftime('%A')
                return {
                    "date": date_obj.date().isoformat(),
                    "weekday": weekday
                }
            except ValueError:
                return {"error": "Ошибка при разборе даты"}
        else:
            return {"error": "Дата не найдена"}
    return {"error": "OCR не распознал текст"}

@router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/x-matroska"]:
        raise HTTPException(status_code=400, detail="Invalid video format")

    video_id = str(uuid.uuid4())
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Подсчет людей
    counter = AsyncPersonCounter(video_path)
    unique_count = await counter.count_people()

    # Получение первого кадра и извлечение даты
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if success:
        timestamp_info = extract_timestamp_easyocr(frame)
    else:
        timestamp_info = {"error": "Не удалось прочитать кадр"}

    # Сохраняем результат
    result_data = {
        "total_people": unique_count,
        "timestamp_info": timestamp_info
    }

    result_path = os.path.join(RESULTS_DIR, f"{video_id}_result.json")
    with open(result_path, 'w') as result_file:
        json.dump(result_data, result_file, ensure_ascii=False, indent=2)

    return JSONResponse(content=result_data)
