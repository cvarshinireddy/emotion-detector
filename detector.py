import cv2
from fer.fer import FER

class EmotionDetector:
    def __init__(self):
        self.detector = FER(mtcnn=False, min_face_size=80, min_neighbors=6)
    def analyze(self, frame):
        results = self.detector.detect_emotions(frame)
        return results

    def top_emotion_for(self, emotions_dict):
        label = max(emotions_dict, key=emotions_dict.get)
        return label, emotions_dict[label]
