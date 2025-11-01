import cv2
import pickle
import numpy as np
from utils.hand_detector import HandDetector

# Load trained model
model_path = "model/sign_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize hand detector
detector = HandDetector()

def main():
    cap = cv2.VideoCapture(0)
    print("📷 Starting live sign detection... (Press 'q' to quit)")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect and draw hands
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)

        # Predict only if landmarks are found
        if lmList:
            flattened = []
            for lm in lmList:
                if isinstance(lm, (list, tuple)) and len(lm) >= 3:
                    flattened.extend(lm[1:3])

            if len(flattened) == 42:  # 21 landmarks × 2 coords
                X = np.array(flattened).reshape(1, -1)
                prediction = model.predict(X)[0]
                prob = model.predict_proba(X).max() * 100

                cv2.putText(frame, f"{prediction} ({prob:.1f}%)",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
