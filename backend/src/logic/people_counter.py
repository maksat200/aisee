import uuid
import cv2
import numpy as np
import asyncio
import os
import json
import tempfile
import time
from typing import Optional, Dict, Tuple, List, Any, Union
from ultralytics import YOLO
import aiohttp
from dataclasses import dataclass, field
import torch
from torchvision.transforms import functional as F
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    """Представление обнаруженного объекта с функциями внешнего вида."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    feature: Optional[np.ndarray] = None
    class_id: int = 0  # ID класса (например, человек)

class KalmanBoxTracker:
    """
    Реализация трекера Калмана для отслеживания и прогнозирования движения ограничивающих рамок.
    Состояние представлено как [x, y, a, h, vx, vy, va, vh], где (x, y) - центр рамки,
    a - соотношение сторон, h - высота, v* - соответствующие скорости.
    """
    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int], feature: Optional[np.ndarray] = None):
        """
        Инициализация трекера с начальным положением и опциональным вектором признаков.
        """
        # Определяем матрицы для фильтра Калмана
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # Больший процессный шум для более быстрой адаптации к изменениям скорости
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        
        # Инициализация состояния
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        self.kf.statePost = np.array([
            [center_x], [center_y], [width / height], [height], [0], [0], [0], [0]
        ], np.float32)
        
        # ID трекера и информация о времени жизни
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        
        # Сохраняем начальный вектор признаков внешнего вида
        self.features = [feature] if feature is not None else []
        self.feature = feature  # Текущий вектор признаков
        
        # Храним историю положений для анализа траектории
        self.history = [bbox]

    def update(self, bbox: Tuple[int, int, int, int], feature: Optional[np.ndarray] = None) -> None:
        """
        Обновляет состояние трекера новым измерением.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        # Обновляем историю положений
        self.history.append(bbox)
        if len(self.history) > 30:  # Ограничиваем длину истории
            self.history = self.history[-30:]
            
        # Обновляем вектор признаков внешнего вида
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 100:  # Ограничиваем количество хранимых признаков
                self.features = self.features[-100:]
            # Вычисляем текущий признак как среднее последних 10 признаков
            self.feature = np.mean(self.features[-10:], axis=0) if len(self.features) >= 10 else feature
        
        # Обновляем фильтр Калмана
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        measurement = np.array([[center_x], [center_y], [width / height], [height]], np.float32)
        self.kf.correct(measurement)

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Прогнозирует следующее положение трекера с помощью фильтра Калмана.
        Возвращает предсказанную ограничивающую рамку (x1, y1, x2, y2).
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Преобразуем прогноз обратно в формат ограничивающей рамки
        state = self.kf.statePost
        center_x = state[0, 0]
        center_y = state[1, 0]
        aspect_ratio = state[2, 0]
        height = state[3, 0]
        width = aspect_ratio * height
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        return (x1, y1, x2, y2)

    def get_state(self) -> Tuple[int, int, int, int]:
        """
        Возвращает текущее состояние трекера в виде ограничивающей рамки.
        """
        state = self.kf.statePost
        center_x = state[0, 0]
        center_y = state[1, 0]
        aspect_ratio = state[2, 0]
        height = state[3, 0]
        width = aspect_ratio * height
        
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        return (x1, y1, x2, y2)


class DeepFeatureExtractor:
    """
    Модель для извлечения эмбеддингов изображений для задачи re-id (повторной идентификации).
    """
    def __init__(self, model_path: Optional[str] = None, use_cuda: bool = True):
        """
        Инициализирует экстрактор признаков.
        
        Аргументы:
            model_path: Путь к предварительно обученной модели (если None, используется ResNet по умолчанию)
            use_cuda: Использовать ли CUDA для вычислений если доступна
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        
        # Загружаем предварительно обученную модель для извлечения признаков
        if model_path is not None and os.path.exists(model_path):
            # Загрузка пользовательской модели
            self.model = torch.load(model_path, map_location=self.device)
        else:
            # В качестве запасного варианта используем ResNet от torchvision
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Удаляем последний классификационный слой для получения эмбеддингов
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.to(self.device)
        self.model.eval()
        
        # Стандартный размер входных данных для модели
        self.input_size = (224, 224)
        
        # Нормализация для ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Предобрабатывает изображение для модели.
        
        Аргументы:
            image: Изображение в формате BGR (как из OpenCV)
            
        Возвращает:
            Тензор, готовый для входа в модель
        """
        # Конвертируем BGR в RGB и изменяем размер
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # Конвертируем в тензор PyTorch и нормализуем
        image_tensor = F.to_tensor(image_resized)
        image_normalized = F.normalize(image_tensor, self.mean, self.std)
        
        return image_normalized.unsqueeze(0)  # Добавляем размерность батча

    @torch.no_grad()
    def extract_features(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """
        Извлекает признаки из обнаруженных областей кадра.
        
        Аргументы:
            frame: Кадр видео
            detections: Список ограничивающих рамок (x1, y1, x2, y2)
            
        Возвращает:
            Список векторов признаков для каждой обнаруженной области
        """
        features = []
        
        if not detections:
            return features
        
        # Создаем батч из всех обнаруженных областей
        batch_tensors = []
        
        for bbox in detections:
            x1, y1, x2, y2 = bbox
            
            # Проверка корректности координат
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                # Если координаты некорректны, используем заполнитель (нулевой вектор)
                features.append(np.zeros(2048, dtype=np.float32))
                continue
                
            # Извлекаем область изображения
            crop = frame[y1:y2, x1:x2]
            
            # Пропускаем слишком маленькие области
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                features.append(np.zeros(2048, dtype=np.float32))
                continue
                
            # Предобрабатываем и добавляем в батч
            tensor = self._preprocess(crop)
            batch_tensors.append(tensor)
        
        if not batch_tensors:
            return [np.zeros(2048, dtype=np.float32) for _ in detections]
        
        # Объединяем в батч и передаем через модель
        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        outputs = self.model(batch)
        
        # Преобразуем выходные данные модели в векторы признаков
        for output in outputs:
            # Сжимаем до одномерного вектора и конвертируем в numpy
            feature = output.squeeze().cpu().numpy()
            # Нормализуем вектор признаков для сравнения косинусным расстоянием
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            features.append(feature)
        
        # Добавляем заполнители для недостающих признаков
        while len(features) < len(detections):
            features.append(np.zeros(2048, dtype=np.float32))
            
        return features


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Вычисляет IoU (Intersection over Union) для двух прямоугольников.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Вычисляет косинусное расстояние между двумя векторами.
    Значения ближе к 0 означают большее сходство, ближе к 2 - меньшее сходство.
    """
    if a is None or b is None:
        return 2.0  # Максимальное косинусное расстояние
        
    if np.all(a == 0) or np.all(b == 0):
        return 2.0
        
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)


class DeepSORTTracker:
    """
    Улучшенный трекер на основе алгоритма DeepSORT, сочетающий визуальные признаки
    с фильтрацией Калмана для точного отслеживания объектов.
    
    Аргументы:
        max_age: Максимальное число кадров, в течение которых трек может быть потерян, прежде чем будет удален
        min_hits: Минимальное число успешных обнаружений для подтверждения трека
        iou_threshold: Минимальное значение IoU для сопоставления детекций и предсказаний
        max_cosine_distance: Максимальное косинусное расстояние для сопоставления по признакам внешнего вида
        nn_budget: Максимальное число образцов для сохранения на один трек
    """
    def __init__(
        self, 
        max_age: int = 70, 
        min_hits: int = 3, 
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.6,
        nn_budget: int = 100,
        feature_weight: float = 0.7
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.feature_weight = feature_weight  # Вес признаков в общей метрике сходства
        
        self.trackers: List[KalmanBoxTracker] = []
        # Множество уже подтвержденных уникальных ID для подсчета
        self.confirmed_ids: set = set()
        # Храним общее количество уникальных людей
        self.unique_count: int = 0
        
        # Тепловая карта для визуализации маршрутов движения
        self.heatmap = None
        self.frame_shape = None

    def reset(self) -> None:
        """Сбрасывает состояние трекера."""
        self.trackers = []
        KalmanBoxTracker.count = 0
        self.confirmed_ids = set()
        self.unique_count = 0
        self.heatmap = None

    def _get_tracks(self) -> List[Tuple[int, Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """
        Возвращает список активных треков в формате (id, bbox, feature).
        """
        return [(t.id, t.get_state(), t.feature) for t in self.trackers if t.time_since_update <= self.max_age]

    def _get_confirmed_tracks(self) -> List[Tuple[int, Tuple[int, int, int, int], Optional[np.ndarray]]]:
        """
        Возвращает список подтвержденных треков (используются для рисования и подсчета).
        """
        return [(t.id, t.get_state(), t.feature) for t in self.trackers 
                if t.hits >= self.min_hits and t.time_since_update <= 1]

    def update(self, detections: List[Detection]) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        Обновляет состояние трекера с новыми детекциями.
        
        Аргументы:
            detections: Список объектов Detection с bbox и features
            
        Возвращает:
            Список кортежей (id, bbox) для отслеживаемых объектов
        """
        # Получаем прогнозы от существующих трекеров
        predicted_boxes = []
        for tracker in self.trackers:
            predicted_box = tracker.predict()
            predicted_boxes.append(predicted_box)
        
        # Если нет детекций, просто обновляем возраст треков и возвращаем текущие подтвержденные
        if not detections:
            # Удаляем треки, которые слишком долго не обновлялись
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            # Обновляем тепловую карту, если она инициализирована
            self._update_heatmap()
            # Возвращаем подтвержденные треки
            return [(t.id, t.get_state()) for t in self.trackers if t.hits >= self.min_hits]
        
        # Выделяем bbox и features из детекций
        detection_boxes = [d.bbox for d in detections]
        detection_features = [d.feature for d in detections]
        
        # Если нет активных треков, создаем новые для всех детекций
        if len(self.trackers) == 0:
            for i, (bbox, feature) in enumerate(zip(detection_boxes, detection_features)):
                tracker = KalmanBoxTracker(bbox, feature)
                self.trackers.append(tracker)
            # Обновляем тепловую карту
            self._update_heatmap()
            return [(t.id, t.get_state()) for t in self.trackers if t.hits >= self.min_hits]
        
        # Строим матрицу стоимости для сопоставления треков с детекциями
        # Комбинируем IoU и косинусное расстояние признаков
        cost_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)
        
        for i, tracker in enumerate(self.trackers):
            for j, (det_box, det_feature) in enumerate(zip(detection_boxes, detection_features)):
                # Вычисляем IoU между предсказанным боксом и детекцией
                iou_cost = 1.0 - compute_iou(predicted_boxes[i], det_box)
                
                # Вычисляем косинусное расстояние между признаками, если они доступны
                feature_cost = 0.0
                if tracker.feature is not None and det_feature is not None:
                    feature_cost = cosine_distance(tracker.feature, det_feature)
                else:
                    feature_cost = 1.0  # Нейтральное значение, если признаки недоступны
                
                # Комбинируем метрики с учетом веса признаков
                cost_matrix[i, j] = (1.0 - self.feature_weight) * iou_cost + self.feature_weight * feature_cost
        
        # Гейтинг: отбрасываем сопоставления с низким IoU
        for i, tracker in enumerate(self.trackers):
            for j, det_box in enumerate(detection_boxes):
                if compute_iou(predicted_boxes[i], det_box) < self.iou_threshold:
                    cost_matrix[i, j] = 1e5  # Большое значение для исключения из сопоставления
        
        # Применяем алгоритм Венгерского (иначе называемый алгоритмом Хунгарии)
        # для минимизации общей стоимости сопоставления
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Отфильтровываем сопоставления с высокой стоимостью
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1.0:  # Пороговое значение для принятия сопоставления
                matches.append((row, col))
        
        # Обработка сопоставленных детекций
        matched_tracker_indices = set([match[0] for match in matches])
        matched_detection_indices = set([match[1] for match in matches])
        
        # Обновляем сопоставленные треки
        for tracker_idx, detection_idx in matches:
            self.trackers[tracker_idx].update(
                detection_boxes[detection_idx], 
                detection_features[detection_idx]
            )
            
            # Если трек достиг порога подтверждения, добавляем его в список уникальных ID
            if (self.trackers[tracker_idx].hits >= self.min_hits and 
                self.trackers[tracker_idx].id not in self.confirmed_ids):
                self.confirmed_ids.add(self.trackers[tracker_idx].id)
                self.unique_count = len(self.confirmed_ids)
        
        # Создаем новые треки для несопоставленных детекций
        for i in range(len(detections)):
            if i not in matched_detection_indices:
                new_tracker = KalmanBoxTracker(detection_boxes[i], detection_features[i])
                self.trackers.append(new_tracker)
        
        # Удаляем треки, которые слишком долго не обновлялись
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        
        # Обновляем тепловую карту
        self._update_heatmap()
        
        # Возвращаем подтвержденные треки для визуализации и анализа
        return [(t.id, t.get_state()) for t in self.trackers if t.hits >= self.min_hits]

    def _update_heatmap(self) -> None:
        """
        Обновляет тепловую карту на основе текущих положений треков.
        """
        if self.frame_shape is None or self.heatmap is None:
            return
            
        # Обновляем тепловую карту на основе подтвержденных треков
        confirmed_tracks = self._get_confirmed_tracks()
        for _, bbox, _ in confirmed_tracks:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Убедимся, что координаты в пределах кадра
            if 0 <= center_x < self.heatmap.shape[1] and 0 <= center_y < self.heatmap.shape[0]:
                # Добавляем гауссово распределение вокруг центра объекта
                cv2.circle(self.heatmap, (center_x, center_y), 10, (0, 0, 10), -1)

    def init_heatmap(self, frame_shape: Tuple[int, int]) -> None:
        """
        Инициализирует тепловую карту для визуализации траекторий.
        
        Аргументы:
            frame_shape: Размер кадра (высота, ширина)
        """
        self.frame_shape = frame_shape
        self.heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)

    def get_heatmap(self) -> np.ndarray:
        """
        Возвращает текущую тепловую карту в цветном формате.
        """
        if self.heatmap is None:
            return None
            
        # Применяем цветовую карту для визуализации
        colored_heatmap = cv2.applyColorMap(self.heatmap, cv2.COLORMAP_JET)
        return colored_heatmap


class AsyncImprovedPeopleCounter:
    """
    Асинхронный класс для подсчёта уникальных людей на видео с использованием DeepSORT.
    
    Поддерживает как локальные файлы, так и URL для видео. При указании URL происходит
    асинхронное скачивание видео во временный файл.
    
    Аргументы:
        video_source (str): Локальный путь к видео или URL.
        source_type (str): 'local' для локального файла или 'url' для ссылки.
        confidence (float): Порог уверенности для детекции.
        show_video (bool): Флаг для отображения видео во время обработки.
        save_video (bool): Флаг для сохранения видео с наложенной информацией.
        output_video_path (Optional[str]): Путь для сохранения видео с результатами.
        results_path (Optional[str]): Путь для сохранения результатов в формате JSON.
        model_path (Optional[str]): Путь к модели YOLO (если None, будет загружена yolov8x.pt).
        feature_extractor (Optional[str]): Путь к модели для извлечения признаков (если None, будет использована модель по умолчанию).
        max_age (int): Максимальное количество кадров для отслеживания потерянных объектов.
        min_hits (int): Минимальное количество детекций для подтверждения трека.
        iou_threshold (float): Порог IoU для сопоставления треков и детекций.
        use_cuda (bool): Использовать ли CUDA для обработки, если доступна.
        skip_frames (int): Сколько кадров пропускать между обработками (для ускорения).
        show_heatmap (bool): Отображать ли тепловую карту движения.
        save_heatmap (bool): Сохранять ли тепловую карту в конце обработки.
        heatmap_path (Optional[str]): Путь для сохранения тепловой карты.
    """
    def __init__(
        self,
        video_source: str,
        source_type: str = "local",
        confidence: float = 0.65,
        show_video: bool = False,
        save_video: bool = True,
        output_video_path: Optional[str] = None,
        results_path: Optional[str] = None,
        model_path: Optional[str] = None,
        feature_extractor: Optional[str] = None,
        max_age: int = 70,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_cuda: bool = True,
        skip_frames: int = 0,
        show_heatmap: bool = False,
        save_heatmap: bool = True,
        heatmap_path: Optional[str] = None,
    ) -> None:
        self.video_source = video_source
        self.source_type = source_type.lower()
        self.confidence = confidence
        self.show_video = show_video
        self.save_video = save_video
        self.output_video_path = output_video_path
        self.results_path = results_path or "results.json"
        self.use_cuda = use_cuda
        self.skip_frames = skip_frames
        self.show_heatmap = show_heatmap
        self.save_heatmap = save_heatmap
        self.heatmap_path = heatmap_path
        
        # Инициализируем модель детекции YOLO
        yolo_model = model_path or "yolov8m.pt"  # Используем более точную модель
        print(f"Initializing YOLO model: {yolo_model}")
        self.model = YOLO(yolo_model)
        
        # Инициализируем экстрактор признаков
        print("Initializing feature extractor")
        self.feature_extractor = DeepFeatureExtractor(feature_extractor, use_cuda)
        
        # Инициализируем трекер DeepSORT
        print("Initializing DeepSORT tracker")
        self.tracker = DeepSORTTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            max_cosine_distance=0.6,
            nn_budget=100,
            feature_weight=0.7,
        )
        
        # Переменные для видео
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: int = 0
        self.out: Optional[cv2.VideoWriter] = None
        self.video_file_path: Optional[str] = None
        
        # Статистика обработки
        self.processing_times: List[float] = []
        self.fps_list: List[float] = []
        self.frame_count: int = 0
        self.total_frames: int = 0

    async def _download_video(self, url: str) -> str:
        """
        Асинхронно скачивает видео по URL и сохраняет во временный файл.
        Возвращает путь к скачанному файлу.
        """
        print(f"Загрузка видео с URL: {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise ValueError(f"Ошибка загрузки видео по URL. Статус: {resp.status}")
                data = await resp.read()
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file.write(data)
                temp_file.close()
                print(f"Видео скачано во временный файл: {temp_file.name}")
                return temp_file.name

    async def _prepare_video(self) -> None:
        """
        Подготавливает видео для обработки: если источник – URL, скачивает видео,
        иначе проверяет локальный файл.
        """
        if self.source_type == "url":
            self.video_file_path = await self._download_video(self.video_source)
        elif self.source_type == "local":
            if not os.path.exists(self.video_source):
                raise ValueError(f"Локальный видеофайл не найден: {self.video_source}")
            self.video_file_path = self.video_source
            print(f"Используется локальный файл: {self.video_file_path}")
        else:
            raise ValueError("source_type должен быть 'local' или 'url'.")

        self.cap = cv2.VideoCapture(self.video_file_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {self.video_file_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Инициализируем тепловую карту
        self.tracker.init_heatmap((self.frame_height, self.frame_width))

        # Подготавливаем запись видео, если необходимо
        if self.save_video:
            if not self.output_video_path:
                base, _ = os.path.splitext(self.video_file_path)
                self.output_video_path = f"{base}_counted_deepsort.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(
                self.output_video_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )
            print(f"Видео с результатами будет сохранено в: {self.output_video_path}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, Tuple[int, int, int, int]]]]:
        """
        Обрабатывает один кадр: проводит детекцию, обновляет трекинг и отрисовывает метки.
        
        Возвращает обработанный кадр и список активных треков.
        """
        start_time = time.time()
        
        # Получаем детекции от YOLO
        results = self.model(frame, classes=0, verbose=False)  # Детектируем только класс "person"
        
        # Извлекаем ограничивающие рамки
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                if conf > self.confidence:
                    # Приводим координаты к целым числам
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    detections.append(Detection(bbox=bbox, confidence=conf))
        
        # Если есть детекции, извлекаем признаки внешнего вида
        if detections:
            # Извлекаем признаки из областей изображения
            features = self.feature_extractor.extract_features(frame, [d.bbox for d in detections])
            
            # Присваиваем извлеченные признаки детекциям
            for i, feature in enumerate(features):
                if i < len(detections):
                    detections[i].feature = feature
        
        # Обновляем трекер с новыми детекциями
        tracked_objects = self.tracker.update(detections)
        
        # Создаем копию кадра для визуализации
        vis_frame = frame.copy()
        
        # Если требуется отображение тепловой карты, накладываем ее на визуализацию
        if self.show_heatmap:
            heatmap = self.tracker.get_heatmap()
            if heatmap is not None:
                # Накладываем тепловую карту с прозрачностью
                alpha = 0.3
                vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, heatmap, alpha, 0)
        
        # Отрисовываем каждый отслеживаемый объект
        for obj_id, bbox in tracked_objects:
            # Используем разные цвета для разных ID
            color = ((obj_id * 37) % 256, (obj_id * 91) % 256, (obj_id * 151) % 256)
            
            # Рисуем ограничивающую рамку
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Добавляем метку с ID
            label = f"ID: {obj_id}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(vis_frame, (bbox[0], bbox[1] - t_size[1] - 10), 
                         (bbox[0] + t_size[0], bbox[1]), color, -1)
            cv2.putText(vis_frame, label, (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Отображаем итоговый счёт уникальных людей и дополнительную информацию
        count_text = f"Уникальных людей: {self.tracker.unique_count}"
        cv2.putText(vis_frame, count_text, (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        
        # Отображаем информацию о производительности
        end_time = time.time()
        processing_time = end_time - start_time
        self.processing_times.append(processing_time)
        
        # Вычисляем среднее время обработки и FPS для последних 30 кадров
        if len(self.processing_times) > 30:
            self.processing_times = self.processing_times[-30:]
        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        self.fps_list.append(avg_fps)
        
        # Отображаем информацию о производительности
        cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Отображаем прогресс обработки
        progress = self.frame_count / self.total_frames if self.total_frames > 0 else 0
        cv2.putText(vis_frame, f"Прогресс: {progress*100:.1f}%", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return vis_frame, tracked_objects

    def _cleanup(self) -> None:
        """
        Освобождает ресурсы: закрывает захват видео, записывающее устройство и окна.
        """
        if self.cap:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        
        # Сохраняем тепловую карту, если требуется
        if self.save_heatmap and self.heatmap_path:
            heatmap = self.tracker.get_heatmap()
            if heatmap is not None:
                cv2.imwrite(self.heatmap_path, heatmap)
                print(f"Тепловая карта сохранена в: {self.heatmap_path}")

    async def run(self) -> Dict[str, Any]:
        """
        Асинхронно обрабатывает видео, применяя детекцию и трекинг к каждому кадру.
        Возвращает результаты обработки.
        """
        await self._prepare_video()
        self.frame_count = 0

        def process_video() -> None:
            skipped_frames = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Пропускаем кадры для ускорения, если установлен skip_frames
                if self.skip_frames > 0 and skipped_frames < self.skip_frames:
                    skipped_frames += 1
                    continue
                skipped_frames = 0
                
                # Обрабатываем кадр и получаем визуализацию и треки
                vis_frame, _ = self.process_frame(frame)
                
                # Отображаем результат, если требуется
                if self.show_video:
                    cv2.imshow("Improved People Counter", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                # Сохраняем результат, если требуется
                if self.out is not None:
                    self.out.write(vis_frame)
                
                # Показываем прогресс каждые 100 кадров
                if self.frame_count % 100 == 0:
                    progress = self.frame_count / self.total_frames * 100 if self.total_frames > 0 else 0
                    print(f"Обработано кадров: {self.frame_count}/{self.total_frames} ({progress:.1f}%)")
            
            self._cleanup()
        
        # Обрабатываем видео в отдельном потоке, чтобы не блокировать асинхронный поток
        await asyncio.to_thread(process_video)
        
        # Анализируем производительность
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            max_time = max(self.processing_times)
            min_time = min(self.processing_times)
            
            print(f"Среднее время обработки кадра: {avg_time*1000:.2f} мс")
            print(f"Минимальное время обработки: {min_time*1000:.2f} мс")
            print(f"Максимальное время обработки: {max_time*1000:.2f} мс")
            print(f"Средний FPS: {1.0/avg_time:.2f}")
        
        print(f"Всего уникальных людей: {self.tracker.unique_count}")
        
        # Формируем результаты
        results = {
            "unique_people": self.tracker.unique_count,
            "processed_frames": self.frame_count,
            "total_frames": self.total_frames,
            "avg_processing_time_ms": avg_time * 1000 if self.processing_times else None,
            "avg_fps": 1.0 / avg_time if self.processing_times and avg_time > 0 else None,
            "output_video_path": self.output_video_path if self.save_video else None,
            "heatmap_path": self.heatmap_path if self.save_heatmap else None,
        }
        
        return results

    def save_results(self) -> None:
        """
        Сохраняет результаты подсчёта в JSON файл.
        """
        results = {
            "unique_people": self.tracker.unique_count,
            "total_frames": self.frame_count,
            "avg_fps": sum(self.fps_list) / len(self.fps_list) if self.fps_list else 0,
            "method": "DeepSORT with appearance features and Kalman filtering",
            "detection_model": "YOLOv8l",
            "confidence_threshold": self.confidence,
        }
        
        # Создаем директорию для результатов, если она не существует
        os.makedirs(os.path.dirname(os.path.abspath(self.results_path)), exist_ok=True)
        
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Результаты сохранены в {self.results_path}")


# Пример использования
if __name__ == "__main__":
    async def main():
        input_video = "src/assets/video/walking2_6ppl.mp4"  # Укажите ваш путь к видео
        current_uuid = str(uuid.uuid4())
        
        # Создаем директории для результатов, если они не существуют
        os.makedirs("src/assets/results/video", exist_ok=True)
        os.makedirs("src/assets/results/json", exist_ok=True)
        os.makedirs("src/assets/results/heatmap", exist_ok=True)
        
        counter = AsyncImprovedPeopleCounter(
            video_source=input_video,
            source_type="local",
            confidence=0.7,  # Повышенный порог уверенности для более точных детекций
            show_video=False,
            save_video=False,
            output_video_path=f"src/assets/results/video/{current_uuid}.mp4",
            results_path=f"src/assets/results/json/{current_uuid}.json",
            model_path="yolov8m.pt",  # Используем более точную модель
            max_age=70,  # Увеличенное значение для отслеживания людей при временных окклюзиях
            min_hits=3,  # Минимальное количество детекций для подтверждения трека
            iou_threshold=0.3,
            use_cuda=True,
            skip_frames=3,  # Обрабатываем каждый кадр для максимальной точности
            show_heatmap=True,
            save_heatmap=True,
            heatmap_path=f"src/assets/results/heatmap/{current_uuid}.jpg",
        )
        
        results = await counter.run()
        counter.save_results()
        print(results)

    asyncio.run(main())