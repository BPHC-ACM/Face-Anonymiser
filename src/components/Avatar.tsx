import { useEffect, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

export function Avatar() {
  // Load the model
  const { scene } = useGLTF("/avatars/avatar-2.glb");
  const avatarRef = useRef<THREE.Group>(null);

  useEffect(() => {
    // Traverse the model to hide the body
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        // Only keep these parts visible
        const keepList = [
          "Wolf3D_Head",
          "Wolf3D_Teeth",
          "Wolf3D_Beard",
          "Wolf3D_Glasses",
          "Wolf3D_Headwear",
          "EyeLeft",
          "EyeRight",
        ];

        const name = child.name;
        const shouldKeep = keepList.some((part) => name.includes(part));

        // Hide everything else
        if (!shouldKeep) {
          child.visible = false;
        }
      }
    });
  }, [scene]);

  // Position adjusted to center the head (since body is hidden)
  return <primitive object={scene} ref={avatarRef} position={[0, -1.7, 0]} />;
}
