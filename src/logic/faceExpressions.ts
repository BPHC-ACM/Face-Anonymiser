import * as THREE from "three";
import { lerp } from "../utils/math";

export const updateFaceExpressions = (
  scene: THREE.Group,
  blendshapes: any[]
) => {
  const nameMapping: Record<string, string> = {
    jawOpen: "mouthOpen",
    mouthSmileLeft: "mouthSmile",
    mouthSmileRight: "mouthSmile",
    mouthFunnel: "mouthOpen",
    mouthPucker: "mouthOpen",
  };

  // 1. Calculate Targets First (Avoid Overwriting)
  // We store the target value for each Avatar Shape here
  const targetInfluences: Record<string, number> = {};

  blendshapes.forEach((shape) => {
    let targetName = shape.categoryName;

    // Check mapping
    if (nameMapping[targetName]) {
      targetName = nameMapping[targetName];
    }

    // Initialize if not exists
    if (targetInfluences[targetName] === undefined) {
      targetInfluences[targetName] = 0;
    }

    // KEY FIX: Use Math.max
    // If jawOpen is 0.8 and mouthFunnel is 0.0, we keep 0.8
    targetInfluences[targetName] = Math.max(
      targetInfluences[targetName],
      shape.score
    );
  });

  // 2. Apply to Avatar
  scene.traverse((child) => {
    const mesh = child as THREE.SkinnedMesh;

    if (
      mesh.isSkinnedMesh &&
      mesh.morphTargetDictionary &&
      mesh.morphTargetInfluences
    ) {
      // Loop through the targets we calculated above
      Object.keys(targetInfluences).forEach((key) => {
        const index = mesh.morphTargetDictionary![key];

        if (index !== undefined) {
          const current = mesh.morphTargetInfluences![index];
          const target = targetInfluences[key];

          // Apply smoothing
          mesh.morphTargetInfluences![index] = lerp(current, target, 0.5);
        }
      });
    }
  });
};
