import * as THREE from "three";
import { lerp } from "../utils/math";

export const updateFaceExpressions = (
  scene: THREE.Group,
  blendshapes: any[],
  nodes: any // <--- NEW: We need 'nodes' to find the Jaw Bone
) => {
  const nameMapping: Record<string, string> = {
    jawOpen: "mouthOpen",
    mouthSmileLeft: "mouthSmile",
    mouthSmileRight: "mouthSmile",
    mouthFunnel: "mouthOpen",
    mouthPucker: "mouthOpen",
  };

  // 1. Calculate Targets
  const targetInfluences: Record<string, number> = {};

  // Track Jaw Openness specifically for the Bone Logic
  let jawOpenScore = 0;

  blendshapes.forEach((shape) => {
    const rawName = shape.categoryName;
    let score = shape.score;

    // SENSITIVITY BOOST:
    // If it's the jaw, multiply it by 3 so it opens easier
    if (rawName === "jawOpen") {
      score *= 3.0;
      score = Math.min(score, 1.0); // Clamp to max 1.0
      jawOpenScore = score; // Save for bone rotation
    }

    // Map to Avatar Name
    let targetName = rawName;
    if (nameMapping[rawName]) {
      targetName = nameMapping[rawName];
    }

    // Initialize & Apply Max Logic
    if (targetInfluences[targetName] === undefined) {
      targetInfluences[targetName] = 0;
    }
    targetInfluences[targetName] = Math.max(
      targetInfluences[targetName],
      score
    );
  });

  // 2. Apply Morph Targets (The Skin)
  scene.traverse((child) => {
    const mesh = child as THREE.SkinnedMesh;
    if (
      mesh.isSkinnedMesh &&
      mesh.morphTargetDictionary &&
      mesh.morphTargetInfluences
    ) {
      Object.keys(targetInfluences).forEach((key) => {
        const index = mesh.morphTargetDictionary![key];
        if (index !== undefined) {
          const current = mesh.morphTargetInfluences![index];
          const target = targetInfluences[key];
          mesh.morphTargetInfluences![index] = lerp(current, target, 0.5);
        }
      });
    }
  });

  // 3. Apply Jaw Bone Rotation (The Skeleton)
  // This is a fallback. If the morph target 'mouthOpen' is empty,
  // this physically rotates the jaw bone.
  const jawBone = nodes.Jaw || nodes.Wolf3D_Jaw; // Common names for RPM jaw
  if (jawBone) {
    const currentX = jawBone.rotation.x;
    // 0.2 is roughly 12 degrees of rotation, which is a wide open mouth
    const targetX = jawOpenScore * 0.2;

    // Smoothly rotate the jaw
    jawBone.rotation.x = lerp(currentX, targetX, 0.5);
  }
};
