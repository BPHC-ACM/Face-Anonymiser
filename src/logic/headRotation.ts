import * as THREE from "three";
import { dampEuler } from "../utils/math";

export const updateHeadRotation = (
  nodes: any,
  matrix: any,
  smoothRotation: THREE.Euler
) => {
  // Convert Matrix to Euler Angles
  const rotationMatrix = new THREE.Matrix4().fromArray(matrix.data);
  const targetEuler = new THREE.Euler().setFromRotationMatrix(rotationMatrix);

  // Smooth the rotation 
  const damped = dampEuler(smoothRotation, targetEuler, 0.2);
  smoothRotation.copy(damped);

  // Find the Bone
  const headBone = (nodes.Head || nodes.Neck) as THREE.Bone;

  // Apply Rotation with Scale
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
