# Face-Anonymizer

# Face-Anonymizer

## Project Description
**Face-Anonymiser** is a Python-based application designed to anonymize faces in real-time video streams using computer vision and deep learning techniques. It leverages MediaPipe for facial landmark detection and OpenCV for video processing. The application allows users to replace detected faces with avatars, which can be switched using hand-swipe gestures. The project also includes notebook-based workflows for advanced avatar generation and integration with ComfyUI, a UI for generative models.

## Technical Specification
### Core Features
- Real-time Face Detection using MediaPipe Face Mesh
- Avatar Replacement with gesture-based switching
- Gesture Recognition via MediaPipe Hands
- Avatar Gallery loaded from `face-avatars` directory
- Video Processing with OpenCV
- Advanced workflows in Jupyter Notebooks for avatar/model generation
- **Flow Matching Enhancement:** Incorporates a generative model for flow matching, enabling seamless and realistic blending of avatars onto faces by learning pixel-wise correspondences and transformations. This improves the quality and naturalness of anonymization, especially in dynamic scenes.

### File Structure
- `anon.py`: Main application script
- `face_mesh.py`: Face mesh logic
- `face_mesh.ipynb`: Notebook for face mesh experimentation
- `face-avatars/`: Avatar images
- `notebooks/comfyui_colab_with_manager.ipynb`: Advanced avatar/model workflows
- `README.md`: Project documentation

### Key Algorithms
- Facial Landmark Detection (468 points per frame)
- Delaunay Triangulation for warping avatars
- Gesture Detection for avatar switching
- Image Warping using OpenCV and NumPy
- **Flow Matching Generative Model:** Learns and applies pixel-wise flow fields to blend avatars onto faces, enhancing realism and adaptability to facial movements.

## Workflow
1. Initialization: Load models and avatars
2. Video Capture: Start webcam stream
3. Face Detection & Landmark Extraction
4. Gesture Recognition for avatar switching
5. Avatar Warping & Replacement
6. **Flow Matching:** Apply generative flow matching model to blend avatars onto faces for improved realism
7. Display Output: Show anonymized video
8. Advanced Avatar Generation (optional, via notebooks)

## Tech Stack
- Python 3.12+
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Torch (`torch`, `torchvision`)
- Pillow (`PIL`)
- Matplotlib
- Other packages for avatar/model generation: `manifold3d`, `pycollada`, `svg-path`, etc.
- Jupyter Notebooks
- ComfyUI
- **Flow Matching Model:** Deep learning model for pixel-wise flow estimation and blending (can be implemented in PyTorch or TensorFlow)

## Example Avatar Gallery
- `face-avatars/character1.jpg`
- `face-avatars/character2.png`
- Supports more avatars as needed

## Integration & Extensibility
- Avatar Generation: Custom avatars via generative models in notebooks
- Model Management: ComfyUI integration
- Cloud/Colab Support: Notebooks for cloud workflows
- **Flow Matching:** Easily extendable to use state-of-the-art generative flow models for improved anonymization quality

## Summary
Face-Anonymiser is a modular, extensible tool for anonymizing faces in video streams using avatars. It combines real-time computer vision, gesture recognition, generative AI workflows, and now flow matching for enhanced realism, making it suitable for privacy applications, creative projects, and research.

