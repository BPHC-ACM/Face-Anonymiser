import cv2
import mediapipe as mp
import numpy as np

import os
import time

# --- 1. Initialization and Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# To change avatars by a simple hand-swipe gesture
mp_hands = mp.solutions.hands  
SWIPE_COOLDOWN = 1.5  # Cooldown between each successive wsipe


# Open camera stream
cap = cv2.VideoCapture(0)


# --- 2. Data Loading & Pre-processing ---

def get_delaunay_triangulation_data(frame, face_landmarks):
    h, w, _ = frame.shape
    points = []
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    points_np = np.array(points, dtype=np.int32)
    
    hull_indices = cv2.convexHull(points_np, returnPoints=False)
    hull_points = points_np[hull_indices.flatten()]
    rect = cv2.boundingRect(hull_points)
    
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)
        
    triangle_list = subdiv.getTriangleList()
    delaunay_triangles_indices = []
    
    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        is_inside = lambda p: (rect[0] <= p[0] < rect[0] + rect[2] and 
                               rect[1] <= p[1] < rect[1] + rect[3])
        
        if is_inside(pt1) and is_inside(pt2) and is_inside(pt3):
            idx = []
            for pt in [pt1, pt2, pt3]:
                distances = np.sum((points_np - pt)**2, axis=1)
                landmark_idx = np.argmin(distances)
                idx.append(landmark_idx.item())

            if len(idx) == 3 and idx[0] != idx[1] and idx[1] != idx[2] and idx[0] != idx[2]:
                delaunay_triangles_indices.append(idx)
                
    return points, delaunay_triangles_indices


# Function to load all avatars 
def load_avatar_gallery(folder_path):
    gallery = []
    print(f"Loading avatars from '{folder_path}'...")
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder '{folder_path}' not found.")
        return []

    # Initialize a temporary FaceMesh just for pre-processing avatars
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as avatar_mesh:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder_path, filename)
                img = cv2.imread(path)
                if img is None: continue

                rgb_avatar = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = avatar_mesh.process(rgb_avatar)

                if results.multi_face_landmarks:
                    print(f" - Processed: {filename}")
                    face_landmarks = results.multi_face_landmarks[0]
                    points, tri_indices = get_delaunay_triangulation_data(img, face_landmarks)
                    
                    # Store everything needed for this specific avatar
                    gallery.append({
                        "name": filename,
                        "image": img,
                        "points": points,
                        "tri_indices": tri_indices
                    })
                else:
                    print(f" - Skipped (No face found): {filename}")
    
    print(f"Successfully loaded {len(gallery)} avatars.")
    return gallery


# Loading the gallery
avatar_gallery = load_avatar_gallery("face-avatars")
if not avatar_gallery:
    print("No valid avatars found. Exiting.")
    exit()


# Keeping track of avatar in place
avatar_index = 0

# --- 3. Warping Helper Functions ---

def get_sub_image_coords(img, points):
    points_np = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points_np)
    points_rel = [(p[0] - x, p[1] - y) for p in points]
    cropped_img = img[y:y+h, x:x+w].copy()
    return cropped_img, points_rel, (x, y, w, h)

def warp_triangle(img_source, img_dest, tri_source_pts, tri_dest_pts):
    cropped_source, tri_source_rel, rect_source = get_sub_image_coords(img_source, tri_source_pts)
    cropped_dest, tri_dest_rel, rect_dest = get_sub_image_coords(img_dest, tri_dest_pts)
    
    tri_source_rel_np = np.float32(tri_source_rel)
    tri_dest_rel_np = np.float32(tri_dest_rel)
    
    M = cv2.getAffineTransform(tri_source_rel_np, tri_dest_rel_np)
    size_dest = (rect_dest[2], rect_dest[3])
    warped_triangle = cv2.warpAffine(cropped_source, M, size_dest, None, 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_REFLECT_101)
    
    mask = np.zeros((rect_dest[3], rect_dest[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, tri_dest_rel_np.astype(np.int32), (255, 255, 255))
    
    mask_inv = cv2.bitwise_not(mask)
    img_dest_masked = cv2.bitwise_and(cropped_dest, mask_inv)
    warped_triangle_masked = cv2.bitwise_and(warped_triangle, mask)
    final_output = cv2.add(img_dest_masked, warped_triangle_masked)
    
    x_dest, y_dest, w_dest, h_dest = rect_dest
    img_dest[y_dest:y_dest+h_dest, x_dest:x_dest+w_dest] = final_output

def complete_warping_pipeline(frame_user, frame_avatar, points_user, points_avatar, tri_indices_list):
    warped_face = np.zeros_like(frame_user, dtype=frame_user.dtype)
    for indices in tri_indices_list:
        tri_user_pts = [points_user[i] for i in indices]
        tri_avatar_pts = [points_avatar[i] for i in indices]
        warp_triangle(frame_avatar, warped_face, tri_avatar_pts, tri_user_pts)
    return warped_face

# --- 4. Main Video Loop with Hand Tracking ---

print("Starting video stream... Swipe hand to change avatar!")

# Initialize Hand tracking and Face Mesh
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh, \
     mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    last_swipe_time = 0
    
    while cap.isOpened():
        ret, frame_user = cap.read()
        if not ret: continue

        frame_user = cv2.flip(frame_user, 1)
        rgb_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)
    
        results_user = face_mesh.process(rgb_user)
        
        # Processing hands
        results_hands = hands.process(rgb_user)
        
        output_frame = frame_user.copy()

        # Swiping Logic
        if results_hands.multi_hand_landmarks:

            hand_x = results_hands.multi_hand_landmarks[0].landmark[0].x
            
            current_time = time.time()
            if current_time - last_swipe_time > SWIPE_COOLDOWN:
                if hand_x < 0.2: # Left side of screen
                    avatar_index = (avatar_index - 1) % len(avatar_gallery)
                    last_swipe_time = current_time
                    print(f"Swiped Left! Avatar: {avatar_gallery[avatar_index]['name']}")
                elif hand_x > 0.8: # Right side of screen
                    avatar_index = (avatar_index + 1) % len(avatar_gallery)
                    last_swipe_time = current_time
                    print(f"Swiped Right! Avatar: {avatar_gallery[avatar_index]['name']}")

            # Draw hands (optional, helps debugging)
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Face Swapping logic
        if results_user.multi_face_landmarks:
            face_landmarks_user = results_user.multi_face_landmarks[0]
            
            # Get Current Avatar Data
            current_avatar = avatar_gallery[avatar_index]
            
            # Recalculate User Delaunay (Geometry)
            points_user, _ = get_delaunay_triangulation_data(frame_user, face_landmarks_user)
            
            if points_user:
                # Perform Warping with CURRENT avatar data
                warped_face = complete_warping_pipeline(
                    frame_user, 
                    current_avatar['image'], 
                    points_user, 
                    current_avatar['points'], 
                    current_avatar['tri_indices']
                )
                
                # Blending
                points_np_user = np.array(points_user, dtype=np.int32)
                hull_indices_user = cv2.convexHull(points_np_user, returnPoints=False)
                hull_points_user = points_np_user[hull_indices_user.flatten()]
                
                mask = np.zeros_like(frame_user, dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull_points_user, (255, 255, 255))
                
                mask_inv = cv2.bitwise_not(mask)
                output_frame = cv2.bitwise_and(frame_user, mask_inv)
                warped_face_masked = cv2.bitwise_and(warped_face, mask)
                output_frame = cv2.add(output_frame, warped_face_masked)

        # UI: Display current avatar name
        cv2.putText(output_frame, f"Avatar: {avatar_gallery[avatar_index]['name']}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Swipe to Swap Face", output_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break

        if cv2.getWindowProperty("Swipe to Swap Face" , cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()