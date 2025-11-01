import cv2
from utils.hand_detector import HandDetector

def main():
    # Initialize webcam and hand detector
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    print("🖐 Hand detector running... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not access webcam.")
            break

        # Detect and draw hands
        landmarks = detector.get_landmarks(frame)
        frame = detector.draw_landmarks(frame, landmarks)

        # Show video output
        cv2.imshow("Hand Detector", frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
