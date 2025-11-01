import cv2
import os
from utils.hand_detector import HandDetector
from utils.preprocessing import Preprocessor

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Initialize helper classes
    detector = HandDetector()
    preprocessor = Preprocessor(save_dir="data")

    # Ask user for gesture label
    label = input("Enter gesture label (e.g., A, B, Hello): ")

    # Count how many samples already exist for this label
    label_path = os.path.join("data", f"{label}.json")
    sample_count = 0
    if os.path.exists(label_path):
        sample_count = len(open(label_path, "r").readlines())

    print(f"Recording samples for gesture: {label}")
    print("Press 's' to save sample, 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        # Detect hands
        frame = detector.findHands(frame)
        landmarks = detector.findPosition(frame)

        # Show coordinates in terminal and screen
        if landmarks:
            print(f"Landmarks ({len(landmarks)} points): {landmarks}")
            x, y = landmarks[4][1], landmarks[4][2]  # thumb tip example
            cv2.putText(frame, f"Thumb: ({x},{y})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show count of saved samples
        cv2.putText(frame, f"Samples saved: {sample_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display webcam feed
        cv2.imshow("Sign Language Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        # Save landmarks when 's' is pressed
        if key == ord('s'):
            if landmarks:
                preprocessor.save_landmarks(label, landmarks)
                sample_count += 1
                print(f"Saved sample {sample_count} for label '{label}'")
            else:
                print("No hand detected, try again!")

        # Quit when 'q' is pressed
        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
