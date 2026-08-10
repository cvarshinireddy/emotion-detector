import cv2
import time
from detector import EmotionDetector
detector = EmotionDetector()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    results = detector.analyze(frame)
    elapsed = time.time() - start
    for face in results:
        x, y, w, h = face["box"]
        label, conf = detector.top_emotion_for(face["emotions"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} {conf*100:.0f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"{elapsed*1000:.0f} ms/frame", (10, 30),
             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Emotion Detection - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()