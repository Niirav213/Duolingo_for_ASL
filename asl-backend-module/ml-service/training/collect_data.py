"""Data collection script for ASL gesture dataset."""
import cv2
import os
import time
from collections import defaultdict
import mediapipe as mp


class DataCollector:
    """Collect training data for ASL gestures."""

    def __init__(self, output_dir: str = "data/raw"):
        """Initialize data collector."""
        self.output_dir = output_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.class_labels = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "SPACE", "DELETE"
        ]

    def create_directories(self):
        """Create output directories for each class."""
        for gesture in self.class_labels:
            gesture_dir = os.path.join(self.output_dir, gesture)
            os.makedirs(gesture_dir, exist_ok=True)

    def collect_gesture(self, gesture: str, samples_per_gesture: int = 100):
        """Collect samples for a specific gesture."""
        gesture_dir = os.path.join(self.output_dir, gesture)
        cap = cv2.VideoCapture(0)

        print(f"\nCollecting data for gesture: {gesture}")
        print(f"Press 's' to start/stop, 'n' for next gesture, 'q' to quit")
        print(f"Collect {samples_per_gesture} samples")

        collected = 0
        recording = False
        frame_count = 0

        while collected < samples_per_gesture:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Run hand detection
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

            # Display status
            status_text = f"Recording: {collected}/{samples_per_gesture}" if recording else "Press 's' to record"
            cv2.putText(
                frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.putText(
                frame, status_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (0, 0, 255), 2
            )

            cv2.imshow("Data Collection - Press 's' to record", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                recording = not recording
                frame_count = 0
            elif key == ord("n"):
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return False

            if recording and results.multi_hand_landmarks:
                frame_count += 1
                if frame_count % 3 == 0:  # Save every 3rd frame
                    filename = os.path.join(gesture_dir, f"{gesture}_{collected:04d}.jpg")
                    cv2.imwrite(filename, frame)
                    collected += 1
                    print(f"  Collected: {collected}")

        cap.release()
        cv2.destroyAllWindows()
        return True

    def collect_all(self, samples_per_gesture: int = 100):
        """Collect samples for all gestures."""
        self.create_directories()

        for gesture in self.class_labels:
            if not self.collect_gesture(gesture, samples_per_gesture):
                break

        print("\nData collection complete!")


def main():
    """Main collection function."""
    collector = DataCollector(output_dir="../data/raw")

    print("ASL Gesture Data Collection")
    print("=" * 40)
    print(f"Gestures to collect: {', '.join(collector.class_labels)}")
    print()

    samples = int(input("Enter samples per gesture (default 100): ") or "100")
    collector.collect_all(samples_per_gesture=samples)


if __name__ == "__main__":
    main()
