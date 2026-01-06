import cv2
import time
from face_tracker import FaceTracker

def main():
    # 1. Initialize Video Capture (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 2. Initialize Face Tracker
    # refine_landmarks=True ensures we get iris tracking for gaze
    tracker = FaceTracker(refine_landmarks=True)

    print("Starting Face Anonymizer... Press 'q' to quit.")

    prev_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # 3. Detect Landmarks
        # This is where the magic happens - getting the geometry of the face
        results = tracker.process_frame(frame)

        # 4. Visualize
        # Draw the mesh on top of the frame so we can see what's happening
        frame = tracker.draw_landmarks(frame, results)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Show the result
        cv2.imshow('Face Anonymizer - Phase 1: Landmarks', frame)

        # Exit on 'q' key or window close
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Face Anonymizer - Phase 1: Landmarks', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
