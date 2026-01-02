import * as THREE from "three";
import { lerp } from "../utils/math";

export const updateFaceExpressions = (
  scene: THREE.Group,
  blendshapes: any[]
) => {
  // 1. Find all meshes that can be morphed (Head, Teeth, Beard, etc.)
  scene.traverse((child) => {
    // TYPE FIX: We cast child to SkinnedMesh to check for morph targets
    const mesh = child as THREE.SkinnedMesh;

    // Only proceed if it is a Mesh and HAS morph targets
    if (
      mesh.isSkinnedMesh &&
      mesh.morphTargetDictionary &&
      mesh.morphTargetInfluences
    ) {
      // 2. Loop through every blendshape from MediaPipe
      blendshapes.forEach((shape) => {
        const index = mesh.morphTargetDictionary![shape.categoryName];

        if (index !== undefined) {
          // 3. Apply Smoothing (Lerp)
          const current = mesh.morphTargetInfluences![index];
          const target = shape.score;

          // 0.5 = Smooth but responsive
          mesh.morphTargetInfluences![index] = lerp(current, target, 0.5);
        }
      });
    }
  });
};
