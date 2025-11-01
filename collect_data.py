import cv2
from utils.hand_detector import HandDetector
from utils.preprocessing import Preprocessor

# Initialize helper classes
detector = HandDetector()
preprocessor = Preprocessor(save_dir="data")

def main():
    cap = cv2.VideoCapture(0)  # open webcam
    label = input("Enter label for this sign (e.g. A, B, Hello): ")

    print("\nPress 's' to save this frame's landmarks")
    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect hands and landmarks
        landmarks = detector.get_landmarks(frame)

        # Draw landmarks on the frame
        frame = detector.draw_landmarks(frame, landmarks)

        # Show the video feed
        cv2.imshow("Sign Language Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        # Save landmarks on 's'
        if key == ord('s'):
            if landmarks:
                preprocessor.save_landmarks(label, landmarks)
                print(f"✅ Saved landmarks for label '{label}'")
            else:
                print("⚠️ No hand detected, try again!")

        # Quit on 'q'
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
