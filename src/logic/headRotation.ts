// src/utils/headRotation.ts (or wherever the file is)
import * as THREE from "three";
import { dampEuler } from "../utils/math";

export const updateHeadRotation = (
  nodes: any,
  matrix: any,
  smoothRotation: THREE.Euler
) => {
  const rotationMatrix = new THREE.Matrix4().fromArray(matrix.data);
  const targetEuler = new THREE.Euler().setFromRotationMatrix(rotationMatrix);

  const damped = dampEuler(smoothRotation, targetEuler, 0.2);
  smoothRotation.copy(damped);

  const headBone = (nodes.Head || nodes.Neck) as THREE.Bone;

  const rotationScale = { x: 1, y: 1.0, z: 1 };

  if (headBone) {
    headBone.rotation.set(
      // Pitch (up/down) – keep as‑is
      damped.x * rotationScale.x,
      // Yaw (left/right) – invert to match webcam orientation
      -damped.y * rotationScale.y,
      // Roll (ear to shoulder) – keep as‑is
      damped.z * rotationScale.z
    );
  }
};
