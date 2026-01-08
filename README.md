# SAFE-FM: Secure, Anonymous Face Editing using Flow Matching

**SAFE-FM** is a real-time face anonymization system that replaces a user's identity with a target avatar while preserving facial expressions, eye gaze, and head pose.

Unlike traditional methods that simply blur the face (destroying utility) or swap faces using GANs (often unstable), this project utilizes **Flow Matching**—a modern generative AI technique that is faster and more stable than standard Diffusion models.

---

## 🎭 Why Flow Matching? (vs. Traditional Avatar Rigging)

When building privacy tools or VTubing avatars, there are generally two approaches. We chose the second one.

### 1. Traditional 3D Avatar Rigging
*   **How it works:** A 3D model (mesh) is "rigged" with virtual bones. The webcam tracks facial points and rotates these bones.
*   **Pros:** Extremely clean edges, runs on very low-end hardware (CPUs), consistent geometry.
*   **Cons:** 
    *   **"Game-like" Appearance:** Looks synthetic/cartoonish unless you have AAA-game assets.
    *   **Expression Loss:** Hard to map subtle 2D lip twitches or eye movements to 3D bones without depth sensors (like FaceID).
    *   **Rigidity:** Can't handle loose hair, lighting changes, or accessories easily.

### 2. SAFE-FM (Our Approach)
*   **How it works:** We treat the video feed as a **Generative** task. We take "Noise" and guide it to hallucinate a new face that matches the geometry of the user's current expression.
*   **Pros:**
    *   **Photo-realism:** Can generate real skin textures, lighting, and shadows.
    *   **Privacy:** The original identity is mathematically destroyed (replaced by noise) before reconstruction.
    *   **Fluidity:** Handles soft deformations (cheeks puffing, lips pursing) naturally.
*   **Cons:** Computationally expensive (requires GPU) and currently lower resolution (128x128 in prototype).

---

## 📂 Project Structure & File Guide

We have built this project from scratch, moving from Vision to Math to Deep Learning.

### 1. The Vision (`face_tracker.py`)
*   **Function:** Wraps Google's **MediaPipe Face Mesh**.
*   **What it does:** Extracts 478 3D landmarks from the webcam feed in real-time.
*   **Key Feature:** We use `refine_landmarks=True` to track Irises, allowing the avatar to look exactly where you look.

### 2. The Math (`flow_solver.py`)
*   **Function:** Implements the **Ordinary Differential Equation (ODE)** solver.
*   **Theory:** In Flow Matching, we define a "Velocity Field" that moves pixels from a random state (Gaussian Noise) to a data state (Face Image) along a straight line.
*   **Method:** We use the **Euler Method**, which iteratively updates the image: $x_{t+1} = x_t + v(x_t, t) 
cdot dt$.

### 3. The Brain (`model.py`)
*   **Function:** A Conditional **U-Net** Architecture.
*   **Inputs:**
    1.  `x`: The current noisy image.
    2.  `t`: The time step (how much noise is left).
    3.  `cond`: The landmark mask (the geometry control signal).
*   **Output:** The "Velocity" (direction) needed to denoise the image towards the target face.

### 4. The Teacher (`train.py`)
*   **Function:** Training script.
*   **Current Mode:** **"One-Shot Adaptation"**. The model learns to map the generic MediaPipe landmarks to a specific target identity (from `face-avatars/`).
*   **Capabilities:** Supports Data Augmentation (Rotation, Scale, Shift) to create a robust model from a single image. Automatically detects CUDA (Nvidia), XPU (Intel), or CPU.
*   **Outcome:** Produces a `model.pth` file containing the weights.

### 5. The Application (`anonymize.py`)
*   **Function:** Real-time inference loop.
*   **Pipeline:** Webcam Capture $\rightarrow$ Extract Landmarks $\rightarrow$ Generate Random Noise $\rightarrow$ Model Predicts Velocity $\rightarrow$ Solver Denoises $\rightarrow$ Display.
*   **Performance:** Runs at ~15-30 FPS on consumer GPUs (depending on step count).

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   NVIDIA GPU (CUDA) or Intel GPU (XPU) recommended.
*   Dependencies: `torch`, `opencv-python`, `mediapipe`, `numpy`.

### Installation
```bash
pip install -r requirements.txt
# If using Intel GPU:
# pip install intel-extension-for-pytorch
```

### Usage
1.  **Train the Model (Local):**
    *   **Check Hardware:** Run `python check_gpu.py` to see if your GPU (CUDA/Intel) is detected.
    *   **Start Training:**
        ```bash
        python train.py
        ```
        *This will train on your local machine. It saves checkpoints to `training_outputs/` every 100 steps. You can stop it anytime with `Ctrl+C`.*

2.  **Run the Live Demo:**
    ```bash
    python anonymize.py
    ```
    *This opens your webcam. The Left side shows the Generated Avatar mimicking you. The Right side shows the Landmark Mask driving the model.*

---

## 🔮 Future Prospects

The current implementation is a **Prototype** (Proof of Concept). To make this a production-grade privacy tool, the following steps are needed:

1.  **Generalization:** Train on a massive dataset (e.g., CelebA-HQ, FFHQ) so the model understands *all* faces, not just one.
2.  **Latent Diffusion:** Move from Pixel Space (128x128) to Latent Space (using a VAE). This allows generating **1024x1024** images with the same computational cost.
3.  **Identity Control:** Add an "Identity Embedding" vector, allowing the user to slide a bar and change their avatar's age, gender, or ethnicity on the fly.
4.  **Temporal Consistency:** Implement "Warm Start" (initializing the next frame using the previous frame's noise) to eliminate flickering.

---
*Created for the SAFE-FM Project, Jan 2026.*