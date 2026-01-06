import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, NamedTuple

class FaceTracker:
    """
    Wrapper for MediaPipe Face Mesh to extract 478 3D facial landmarks.
    This acts as the 'Control' signal for the future Flow Matching model.
    """
    def __init__(self, 
                 max_num_faces: int = 1, 
                 refine_landmarks: bool = True, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            max_num_faces: Maximum faces to detect (we usually want 1 for anonymization).
            refine_landmarks: If True, includes iris landmarks for eye gaze tracking.
            min_detection_confidence: Threshold for the initial face detection.
            min_tracking_confidence: Threshold for tracking to prevent re-detection jitter.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Drawing utilities for visualization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame: np.ndarray) -> Optional[NamedTuple]:
        """
        Processes a BGR video frame and returns the landmark results.
        
        Args:
            frame: A BGR image array from OpenCV.
            
        Returns:
            The raw MediaPipe results object containing .multi_face_landmarks,
            or None if no face is detected.
        """
        # MediaPipe expects RGB images, but OpenCV captures in BGR.
        # We assume the input 'frame' is writeable.
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_frame)
        
        # Restore writeable flag for downstream processing/drawing
        frame.flags.writeable = True
        return results

    def draw_landmarks(self, frame: np.ndarray, results: NamedTuple) -> np.ndarray:
        """
        Draws the face mesh tessellation and contours onto the frame.
        Useful for debugging and verifying tracking quality.
        """
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Draw the mesh tessellation (the net over the face)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # 2. Draw contours (eyes, lips, face oval) - crucial for expression
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
                # 3. Draw Irises (requires refine_landmarks=True)
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )
        return frame
