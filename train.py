import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import random
import time
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from model import FlowMatchingNet
from face_tracker import FaceTracker

# --- Configuration ---
IMG_SIZE = 512       
BATCH_SIZE = 1       # Reduced for Local Training (especially on CPUs/Integrated GPUs)
LEARNING_RATE = 1e-4
TOTAL_STEPS = 5000   
LOG_FREQ = 10        # Log more often locally
SAVE_FREQ = 100      # Save more often locally
AVATAR_PATH = "face-avatars/character1.jpg"
OUTPUT_DIR = "training_outputs"

def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def create_base_data(image_path, tracker):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Avatar image not found at {image_path}")

    print(f"Loading avatar from {image_path}...")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    results = tracker.process_frame(img)
    mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    if results.multi_face_landmarks:
        tracker.draw_landmarks(mask, results)
    else:
        print("WARNING: No face detected in the avatar image! The mask will be empty.")

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    
    return pil_img, pil_mask

@torch.no_grad()
def sample_and_save(net, cond, target_img, step, device):
    net.eval()
    x = torch.randn_like(cond)
    n_steps = 10
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = torch.tensor([i / n_steps], device=device).float()
        v = net(x, t, cond)
        x = x + v * dt

    gen_img = (x.clamp(-1, 1) + 1) / 2
    target_img = (target_img + 1) / 2
    cond_vis = (cond + 1) / 2
    
    grid = torch.cat([cond_vis, gen_img, target_img], dim=3)
    grid_pil = transforms.ToPILImage()(grid.squeeze(0).cpu())
    
    save_path = os.path.join(OUTPUT_DIR, "samples", f"step_{step:05d}.png")
    grid_pil.save(save_path)
    net.train()

def train():
    setup_dirs()
    device = get_device()
    print(f"Training on device: {device}")

    net = FlowMatchingNet().to(device)
    tracker = FaceTracker()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE)

    try:
        base_img, base_mask = create_base_data(AVATAR_PATH, tracker)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Starting Local Training...")
    print("Press Ctrl+C to stop at any time (checkpoints are saved periodically).")
    
    start_time = time.time()
    
    try:
        for step in range(1, TOTAL_STEPS + 1):
            optimizer.zero_grad()
            
            # Augmentation
            angle = random.uniform(-30, 30)
            scale = random.uniform(0.85, 1.15)
            trans_x = random.uniform(-0.1, 0.1) * IMG_SIZE
            trans_y = random.uniform(-0.1, 0.1) * IMG_SIZE
            
            aug_img = TF.affine(base_img, angle, (trans_x, trans_y), scale, 0)
            aug_mask = TF.affine(base_mask, angle, (trans_x, trans_y), scale, 0)
            
            x1 = to_tensor(aug_img).unsqueeze(0).to(device)
            cond = to_tensor(aug_mask).unsqueeze(0).to(device)
            
            # Flow Matching
            t = torch.rand(1, device=device)
            x0 = torch.randn_like(x1)
            t_view = t.view(-1, 1, 1, 1)
            xt = (1 - t_view) * x0 + t_view * x1
            target_v = x1 - x0
            
            pred_v = net(xt, t, cond)
            loss = F.mse_loss(pred_v, target_v)
            
            loss.backward()
            optimizer.step()
            
            if step % LOG_FREQ == 0:
                elapsed = time.time() - start_time
                print(f"Step {step}/{TOTAL_STEPS} | Loss: {loss.item():.6f} | Time: {elapsed:.1f}s")
                
            if step % SAVE_FREQ == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", "model_latest.pth")
                torch.save(net.state_dict(), ckpt_path)
                sample_and_save(net, cond, x1, step, device)
                print(f"Saved checkpoint to {ckpt_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        save = input("Save current model? (y/n): ")
        if save.lower() == 'y':
            ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", "model_interrupted.pth")
            torch.save(net.state_dict(), ckpt_path)
            print(f"Saved model to {ckpt_path}")
            torch.save(net.state_dict(), "model.pth")
            print(f"Saved copy to model.pth")

    print("Training Finished!")
    torch.save(net.state_dict(), "model.pth")
    print("Saved final model to model.pth")

if __name__ == "__main__":
    train()