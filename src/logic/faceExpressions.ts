import * as THREE from "three";
import { lerp } from "../utils/math";

export const updateFaceExpressions = (
  scene: THREE.Group,
  blendshapes: any[]
) => {
  scene.traverse((child) => {
    const mesh = child as THREE.SkinnedMesh;

    // We look for any mesh that has morph targets
    if (
      mesh.isSkinnedMesh &&
      mesh.morphTargetDictionary &&
      mesh.morphTargetInfluences
    ) {
      blendshapes.forEach((shape) => {
        let name = shape.categoryName;
        const score = shape.score;

        // MIRROR FIX
        if (name.includes("Left")) {
          name = name.replace("Left", "Right");
        } else if (name.includes("Right")) {
          name = name.replace("Right", "Left");
        }

        const index = mesh.morphTargetDictionary![name];

        if (index !== undefined) {
          const current = mesh.morphTargetInfluences![index];
          // Smooth the transition
          mesh.morphTargetInfluences![index] = lerp(current, score, 0.5);
        }
      });
    }
  });
};
