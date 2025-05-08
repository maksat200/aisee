import cv2  # Библиотека для работы с видео и изображениями
import torch  # PyTorch – библиотека для глубокого обучения
import asyncio  # Модуль для асинхронного выполнения задач
import datetime  # Для работы с датами и временем
import re  # Регулярные выражения – для поиска даты в тексте
import easyocr  # OCR-библиотека для распознавания текста на изображениях

from concurrent.futures import ThreadPoolExecutor  # Асинхронный запуск блокирующих функций
from ultralytics import YOLO  # Импорт YOLO модели для обнаружения объектов
from deep_sort_realtime.deepsort_tracker import DeepSort  # Трекер для отслеживания объектов между кадрами


class AsyncPersonCounter:
    def __init__(self, video_path: str, model_name: str = 'yolov8n'):
        self.video_path = video_path  # Путь к видеофайлу
        self.model = YOLO(model_name + '.pt')  # Загрузка YOLO-модели
        self.tracker = DeepSort(max_age=30)  # Инициализация трекера с "временем жизни" трека
        self.unique_ids = set()  # Набор уникальных ID отслеживаемых людей
        self.executor = ThreadPoolExecutor(max_workers=2)  # Потоковый исполнитель для фоновых задач

    async def process_frame(self, frame):
        loop = asyncio.get_event_loop()  # Получаем текущий цикл событий
        results = await loop.run_in_executor(self.executor, self.model, frame)  # Выполнение модели в отдельном потоке
        detections = []  # Список обнаруженных объектов

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and box.conf[0] > 0.5:  # Если объект — человек и уверенность > 0.5
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Координаты ограничивающей рамки
                    detections.append(([x1, y1, x2, y2], box.conf[0].item(), 0))  # Добавляем в список для трекера

        tracked_objects = self.tracker.update_tracks(detections, frame=frame)  # Обновление треков
        for track in tracked_objects:
            if track.is_confirmed():  # Только подтверждённые треки
                self.unique_ids.add(track.track_id)  # Добавление уникального ID

        return frame  # Возврат обработанного кадра

    async def count_people(self) -> int:
        cap = cv2.VideoCapture(self.video_path)  # Открытие видеофайла
        loop = asyncio.get_event_loop()

        while cap.isOpened():  # Пока видео открыто
            ret, frame = await loop.run_in_executor(self.executor, cap.read)  # Чтение кадра в фоновом потоке
            if not ret:
                break  # Если нет кадра – выходим
            await self.process_frame(frame)  # Обработка кадра

        cap.release()  # Освобождение видео
        return len(self.unique_ids)  # Возврат количества уникальных людей


def extract_timestamp_from_video(video_path: str):
    cap = cv2.VideoCapture(video_path)  # Открытие видео
    reader = easyocr.Reader(['en'], gpu=False)  # Создание OCR-читалки (без GPU)
    attempts = 0  # Количество попыток
    timestamp_info = None  # Информация о дате

    while attempts < 5:  # Максимум 5 попыток
        ret, frame = cap.read()  # Чтение кадра
        if not ret:
            break
        timestamp_region = frame[0:60, 0:400]  # Область кадра, где, предположительно, отображается дата
        result = reader.readtext(timestamp_region)  # Распознавание текста

        if result:
            raw_text = result[0][1]  # Получение текста
            print(f"OCR Attempt {attempts + 1}: {raw_text}")  # Вывод распознанного текста
            match = re.search(r'(\d{2})[-/](\d{2})[-/](\d{4})', raw_text)  # Поиск шаблона даты
            if match:
                month, day, year = match.groups()  # Извлечение данных
                try:
                    date_obj = datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")  # Парсинг даты
                    weekday = date_obj.strftime('%A')  # День недели
                    timestamp_info = {
                        "date": date_obj.date().isoformat(),  # ISO формат даты
                        "weekday": weekday  # Название дня недели
                    }
                    break  # Прерывание цикла при успешной распаковке
                except ValueError:
                    pass  # Неверный формат даты – продолжаем
        attempts += 1  # Увеличиваем попытки

    cap.release()  # Закрытие видео
    return timestamp_info or {"error": "OCR не распознал дату"}  # Возврат результата
