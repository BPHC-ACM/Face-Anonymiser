import * as THREE from "three";
import { dampEuler } from "../utils/math";

// We pass the graph nodes, the matrix from MediaPipe, and our smoothRotation reference
export const updateHeadRotation = (
  nodes: any,
  matrix: any,
  smoothRotation: THREE.Euler
) => {
  // 1. Convert Matrix to Euler Angles
  const rotationMatrix = new THREE.Matrix4().fromArray(matrix.data);
  const targetEuler = new THREE.Euler().setFromRotationMatrix(rotationMatrix);

  // 2. Smooth the rotation (Damping)
  // We update the 'smoothRotation' object directly so it persists
  const damped = dampEuler(smoothRotation, targetEuler, 0.2);
  smoothRotation.copy(damped);

  // 3. Find the Bone
  const headBone = (nodes.Head || nodes.Neck) as THREE.Bone;

  // 4. Apply Rotation with Scale
  // Pitch (X), Yaw (Y), Roll (Z)
  const rotationScale = { x: 1, y: 1.0, z: 1 };

  if (headBone) {
    headBone.rotation.set(
      damped.x * rotationScale.x,
      -damped.y * rotationScale.y, // Inverted
      -damped.z * rotationScale.z // Inverted
    );
  }
};
