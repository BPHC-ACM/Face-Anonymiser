import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from model import FlowMatchingNet
from flow_solver import FlowSolver
from face_tracker import FaceTracker

# Config
IMG_SIZE = 512  # HD Resolution for Colab/T4
BATCH_SIZE = 4  

def create_landmark_mask(image_path, tracker):
    """
    Loads an image, extracts landmarks, and draws them on a black background.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    results = tracker.process_frame(img)
    mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    if results.multi_face_landmarks:
        tracker.draw_landmarks(mask, results)
    
    # Return as PIL Images for easier torch transforms
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), \
           Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device} | Resolution: {IMG_SIZE}x{IMG_SIZE}")

    net = FlowMatchingNet().to(device)
    # solver = FlowSolver() # Not needed for training loop anymore
    tracker = FaceTracker()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

    target_path = "face-avatars/character1.jpg"
    if not os.path.exists(target_path):
        print(f"Error: {target_path} not found.")
        return

    # Load Base Data (PIL Images)
    pil_img, pil_mask = create_landmark_mask(target_path, tracker)
    
    # Base Transforms (To Tensor + Normalize)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Starting Heavy Duty Training (512x512)...")
    print("Augmentation Enabled. Saving checkpoints every 500 steps.")

    # Train for 5000 steps for high resolution convergence
    for step in range(5001):
        net.train()
        optimizer.zero_grad()
        
        # --- Data Augmentation Pipeline ---
        # 1. Generate Random Parameters
        angle = random.uniform(-30, 30)      # Rotation
        scale = random.uniform(0.8, 1.2)     # Zoom
        translate_x = random.randint(-40, 40) # Shift (Increased for 512px)
        translate_y = random.randint(-40, 40)
        
        # 2. Apply Transform
        aug_img = TF.affine(pil_img, angle, (translate_x, translate_y), scale, 0)
        aug_mask = TF.affine(pil_mask, angle, (translate_x, translate_y), scale, 0)
        
        # 3. Convert to Tensor
        x1 = to_tensor(aug_img).unsqueeze(0).to(device)
        cond = to_tensor(aug_mask).unsqueeze(0).to(device)

        # --- Flow Matching Math ---
        t = torch.rand(1, device=device)
        x0 = torch.randn_like(x1)
        xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        target_vt = x1 - x0
        
        pred_vt = net(xt, t, cond)
        loss = F.mse_loss(pred_vt, target_vt)
        
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.6f}")

        # Save Checkpoint every 500 steps
        if step % 500 == 0:
            torch.save(net.state_dict(), "model.pth")
            print(f"Saved model.pth at step {step}")

    print("Training complete! Download 'model.pth' from the file browser.")

    print("Training complete! Check the 'results' folder to see the progress.")
    
    # Save the weights for the live demo
    torch.save(net.state_dict(), "model.pth")
    print("Saved model weights to model.pth")

if __name__ == "__main__":
    train()
