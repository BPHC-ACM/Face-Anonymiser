import torch
import cv2
import time
import numpy as np
from torchvision import transforms
from model import FlowMatchingNet
from flow_solver import FlowSolver
from face_tracker import FaceTracker

IMG_SIZE = 512

def main():
    # Device selection: Check for CUDA (Nvidia), XPU (Intel), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    
    # 1. Load Model
    # Optimization: Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    net = FlowMatchingNet().to(device)
    if not torch.os.path.exists("model.pth"):
        print("Error: model.pth not found. Please run train_overfit.py first.")
        return
    
    net.load_state_dict(torch.load("model.pth", map_location=device))
    net.eval()
    
    # 2. Setup Utilities
    # PERFORMANCE TUNING: 
    # - Reduced steps from 5 to 3 for speed (1-2 steps might also work for "turbo" mode).
    # - Using INFERENCE_SIZE=256 reduces compute by 4x compared to 512.
    SOLVER_STEPS = 3
    INFERENCE_SIZE = 512
    
    solver = FlowSolver()
    tracker = FaceTracker(refine_landmarks=True)
    cap = cv2.VideoCapture(0)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Temporal Warm-Start: Use the same base noise across frames to prevent flickering
    # We initialize this at INFERENCE_SIZE
    persistent_noise = torch.randn(1, 3, INFERENCE_SIZE, INFERENCE_SIZE).to(device)

    print(f"Live Anonymizer Started. Resolution: {INFERENCE_SIZE}px | Steps: {SOLVER_STEPS}")
    print("Press 'q' to quit.")

    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        
        # 3. Get Landmarks from YOUR face
        # process_frame works best on reasonably high res images
        # We keep the display/processing logic at IMG_SIZE (512) for consistency with training view
        display_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        results = tracker.process_frame(display_frame)
        
        # 4. Create Landmark Mask (Condition)
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if results.multi_face_landmarks:
            tracker.draw_landmarks(mask, results)
            
        # Optimization: Resize mask to lower inference resolution
        mask_small = cv2.resize(mask, (INFERENCE_SIZE, INFERENCE_SIZE))
        
        # 5. Generate Anonymized Face
        cond = transform(mask_small).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            # Use persistent noise instead of generating new noise every frame
            def model_vt(x, t):
                return net(x, t, cond)
            
            # Solve the ODE
            out = solver.solve_euler(persistent_noise, model_vt, steps=SOLVER_STEPS)
            
            # Denormalize
            out = (out.clamp(-1, 1) + 1) / 2
            # Upscale back to 512 for display
            out = torch.nn.functional.interpolate(out, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            
            gen_face = (out.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gen_face = cv2.cvtColor(gen_face, cv2.COLOR_RGB2BGR)

        # 6. Display - Combine and Resize
        # Stack images horizontally: [Generated Face | Landmark Mask]
        combined = np.hstack((gen_face, mask))
        
        # Current resolution is already 512x512, so the combined width is 1024.
        # This is large enough for display.
        combined_large = combined
        display_h, display_w = combined.shape[:2]
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Draw FPS on the image
        cv2.putText(combined_large, f'FPS: {int(fps)}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add labels
        cv2.putText(combined_large, "Generated Identity", (20, display_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined_large, "Landmark Input", (display_w // 2 + 20, display_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("SAFE-FM Prototype: Anonymized Feed", combined_large)

        # 7. Handle Exit Events
        # Check for 'q' key OR if the user clicked the 'X' close button
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty("SAFE-FM Prototype: Anonymized Feed", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
