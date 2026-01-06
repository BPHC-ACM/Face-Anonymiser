import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from model import FlowMatchingNet
from flow_solver import FlowSolver
from face_tracker import FaceTracker

# Config
IMG_SIZE = 128  # Increased from 64 to 128 for better quality

def create_landmark_mask(image_path, tracker):
    """
    Loads an image, extracts landmarks, and draws them on a black background.
    This creates the 'Control' signal the model needs.
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 1. Get landmarks
    results = tracker.process_frame(img)
    
    # 2. Create black canvas
    mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # 3. Draw landmarks on the black canvas
    if results.multi_face_landmarks:
        tracker.draw_landmarks(mask, results)
    
    return img, mask

def train():
    # Device selection: Check for CUDA (Nvidia), XPU (Intel), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # 1. Setup Models
    net = FlowMatchingNet().to(device)
    solver = FlowSolver()
    tracker = FaceTracker()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

    # 2. Prepare Data (Overfitting on character1.jpg)
    target_path = "face-avatars/character1.jpg"
    if not os.path.exists(target_path):
        print(f"Error: {target_path} not found.")
        return

    img_np, mask_np = create_landmark_mask(target_path, tracker)
    
    # Convert to Tensors (Scale to [-1, 1] for better GAN/Diffusion stability)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    x1 = transform(img_np).unsqueeze(0).to(device)    # Target Image (t=1)
    cond = transform(mask_np).unsqueeze(0).to(device) # Landmarks (Condition)

    print(f"Starting 'Overfitting' Training at {IMG_SIZE}x{IMG_SIZE}...")
    print("The model will learn to generate ONE specific face perfectly.")

    # Increased steps because 128x128 is harder to learn than 64x64
    for step in range(1501):
        net.train()
        optimizer.zero_grad()

        # --- Flow Matching Math ---
        # x_t = (1 - t)*x0 + t*x1 (Straight line path)
        # We want to predict the velocity: vt = x1 - x0
        
        t = torch.rand(1, device=device)
        x0 = torch.randn_like(x1) # Pure noise
        
        # Current noisy state at time t
        xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        
        # Target velocity is simply the straight line: (x1 - x0)
        target_vt = x1 - x0
        
        # Model predicts velocity
        pred_vt = net(xt, t, cond)
        
        # Loss: Mean Squared Error between predicted and true velocity
        loss = F.mse_loss(pred_vt, target_vt)
        
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.6f}")
            
            # --- Inference / Sampling ---
            net.eval()
            with torch.no_grad():
                # Start with pure noise
                test_noise = torch.randn_like(x1)
                
                # Define our vector field function using the model
                def model_vt(x, t_val):
                    return net(x, t_val, cond)
                
                # Use our FlowSolver (Euler) to generate the image
                out = solver.solve_euler(test_noise, model_vt, steps=10)
                
                # Denormalize and save
                out = (out.clamp(-1, 1) + 1) / 2
                out_np = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Save progress
                if not os.path.exists("results"): os.makedirs("results")
                cv2.imwrite(f"results/step_{step}.png", cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))

    print("Training complete! Check the 'results' folder to see the progress.")
    
    # Save the weights for the live demo
    torch.save(net.state_dict(), "model.pth")
    print("Saved model weights to model.pth")

if __name__ == "__main__":
    train()
