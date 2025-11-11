import cv2
import mediapipe as mp
import numpy as np

# --- 1. Initialization and Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Open camera stream
cap = cv2.VideoCapture(0)

# Load the static avatar image (Destination/Source of Texture)
# Replace the placeholder with your actual path
AVATAR_IMAGE_PATH = r"C:\Users\ANAGHA\Downloads\pablo-munoz-gomez-zbg-3d-avatar.jpg"
frame_avatar = cv2.imread(AVATAR_IMAGE_PATH)

if frame_avatar is None:
    print(f"ERROR: Could not load avatar image at {AVATAR_IMAGE_PATH}")
    exit()

# Global variables to store avatar data once
points_avatar = []
tri_indices_list = [] # Common list of triangle indices

# --- 2. Delaunay Triangulation Helper Functions ---

def get_delaunay_triangulation_data(frame, face_landmarks):
    
    h, w, _ = frame.shape
    
    # Store all landmark points as a list of (x, y) tuples
    points = []
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    points_np = np.array(points, dtype=np.int32)
    
    # --- Delaunay Calculation ---
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

        # Function to check if a point is inside the bounding rect
        is_inside = lambda p: (rect[0] <= p[0] < rect[0] + rect[2] and 
                               rect[1] <= p[1] < rect[1] + rect[3])
        
        if is_inside(pt1) and is_inside(pt2) and is_inside(pt3):
            idx = []
            
            for pt in [pt1, pt2, pt3]:
                # Find the index of the landmark using minimum distance
                distances = np.sum((points_np - pt)**2, axis=1)
                landmark_idx = np.argmin(distances)
                idx.append(landmark_idx.item())

            # Store the indices only if they map to unique landmarks
            if len(idx) == 3 and idx[0] != idx[1] and idx[1] != idx[2] and idx[0] != idx[2]:
                delaunay_triangles_indices.append(idx)
                
    return points, delaunay_triangles_indices

# --- 3. Affine Warping Helper Functions ---

def get_sub_image_coords(img, points):
    
    points_np = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points_np)
    
    points_rel = [(p[0] - x, p[1] - y) for p in points]

    cropped_img = img[y:y+h, x:x+w].copy() # .copy() is crucial
    
    return cropped_img, points_rel, (x, y, w, h)

def warp_triangle(img_source, img_dest, tri_source_pts, tri_dest_pts):
    
    # 1. Get minimum bounding box and relative points for both triangles
    cropped_source, tri_source_rel, rect_source = get_sub_image_coords(img_source, tri_source_pts)
    cropped_dest, tri_dest_rel, rect_dest = get_sub_image_coords(img_dest, tri_dest_pts)
    
    tri_source_rel_np = np.float32(tri_source_rel)
    tri_dest_rel_np = np.float32(tri_dest_rel)
    
    # 2. Compute Affine Matrix (M)
    M = cv2.getAffineTransform(tri_source_rel_np, tri_dest_rel_np)
    
    # 3. Apply Affine Transformation (Warp)
    size_dest = (rect_dest[2], rect_dest[3]) # (w, h)
    warped_triangle = cv2.warpAffine(cropped_source, M, size_dest, None, 
                                     flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_REFLECT_101)
    
    # 4. Create Mask for the Destination Triangle
    mask = np.zeros((rect_dest[3], rect_dest[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, tri_dest_rel_np.astype(np.int32), (255, 255, 255))
    
    # 5. Combine Warped Triangle with Destination Image
    
    # Mask out the triangle area in the current destination crop
    mask_inv = cv2.bitwise_not(mask)
    img_dest_masked = cv2.bitwise_and(cropped_dest, mask_inv)
    
    # Apply mask to the warped triangle
    warped_triangle_masked = cv2.bitwise_and(warped_triangle, mask)
    
    # Combine the masked layers
    final_output = cv2.add(img_dest_masked, warped_triangle_masked)
    
    # 6. Insert the result back into the original destination image
    x_dest, y_dest, w_dest, h_dest = rect_dest
    img_dest[y_dest:y_dest+h_dest, x_dest:x_dest+w_dest] = final_output

def complete_warping_pipeline(frame_user, frame_avatar, points_user, points_avatar, tri_indices_list):

    # Initialize the blank canvas (same size as user frame)
    warped_face = np.zeros_like(frame_user, dtype=frame_user.dtype)
    
    for indices in tri_indices_list:
        
        # Get actual pixel coordinates for the current triangle
        tri_user_pts = [points_user[i] for i in indices]      # Destination (where it lands)
        tri_avatar_pts = [points_avatar[i] for i in indices]  # Source (where texture comes from)

        # Warp from AVATAR (source) onto the WARPED_FACE canvas (destination)
        warp_triangle(frame_avatar, 
                      warped_face, 
                      tri_avatar_pts, 
                      tri_user_pts)
                      
    return warped_face


# --- 4. Main Execution Setup (Outside the video loop) ---

print("Processing static avatar image...")

# 4a. Process AVATAR image landmarks ONCE
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as avatar_mesh:
    rgb_avatar = cv2.cvtColor(frame_avatar, cv2.COLOR_BGR2RGB)
    results_avatar = avatar_mesh.process(rgb_avatar)
    
    if results_avatar.multi_face_landmarks:
        # Get avatar points and the common set of triangle indices
        face_landmarks_avatar = results_avatar.multi_face_landmarks[0]
        points_avatar, tri_indices_list = get_delaunay_triangulation_data(frame_avatar, face_landmarks_avatar)
        
        print(f"Avatar processing complete. {len(points_avatar)} landmarks detected. {len(tri_indices_list)} triangles defined.")
    else:
        print("ERROR: No face detected in the avatar image. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# --- 5. Main Video Processing Loop ---

print("Starting video stream for face swap...")

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:

    while cap.isOpened():
        ret, frame_user = cap.read()
        if not ret:
            continue

        frame_user = cv2.flip(frame_user, 1) # Flip for selfie view
        rgb_user = cv2.cvtColor(frame_user, cv2.COLOR_BGR2RGB)
        results_user = face_mesh.process(rgb_user)
        
        output_frame = frame_user.copy() # The final frame

        if results_user.multi_face_landmarks:
            face_landmarks_user = results_user.multi_face_landmarks[0]
            
            # 5a. Get User Landmarks (Destination Geometry)
            # NOTE: We use the already computed tri_indices_list from the avatar
            points_user, _ = get_delaunay_triangulation_data(frame_user, face_landmarks_user)
            
            # 5b. Perform Triangle Warping
            if points_user and points_avatar and tri_indices_list:
                
                # The warped_face contains the avatar texture mapped onto the user's face geometry
                warped_face = complete_warping_pipeline(frame_user, frame_avatar, 
                                                        points_user, points_avatar, 
                                                        tri_indices_list)
                
                # 5c. Seamless Blending (Essential final step)
                # For basic functionality, we'll draw the warped face over the user's face.
                # Advanced projects use cv2.seamlessClone (Poisson Blending) here.
                
                # Get the convex hull mask of the user's face for simple blending
                points_np_user = np.array(points_user, dtype=np.int32)
                hull_indices_user = cv2.convexHull(points_np_user, returnPoints=False)
                hull_points_user = points_np_user[hull_indices_user.flatten()]
                
                mask = np.zeros_like(frame_user, dtype=np.uint8)
                cv2.fillConvexPoly(mask, hull_points_user, (255, 255, 255))
                
                # Combine: (User frame * Inverted Mask) + (Warped frame * Mask)
                mask_inv = cv2.bitwise_not(mask)
                output_frame = cv2.bitwise_and(frame_user, mask_inv)
                warped_face_masked = cv2.bitwise_and(warped_face, mask)
                output_frame = cv2.add(output_frame, warped_face_masked)
                
            else:
                output_frame = frame_user
        
        cv2.imshow("Warped Face Swap Output", output_frame)

        # Exit check
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

# --- 6. Cleanup ---
cap.release()
cv2.destroyAllWindows()
