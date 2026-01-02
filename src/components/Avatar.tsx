// import { useEffect } from "react";
// import * as THREE from "three";
import { useAvatarLogic } from "../hooks/useAvatarLogic";

export function Avatar() {
  // 1. Get the scene from your hook
  const { scene, avatarRef } = useAvatarLogic("/avatars/avatar-1.glb");

    // 2. DEBUGGER: Check for Morph Targets
    // EDIT THE READYPLAYERME URL TO GIVE ARKIT AVATAR
//   useEffect(() => {
//     if (!scene) return;

//     const head = scene.getObjectByName("Wolf3D_Head") as THREE.Mesh;

//     console.log("INSPECTING AVATAR...");
//     if (head && head.morphTargetDictionary) {
//       const shapes = Object.keys(head.morphTargetDictionary);
//       console.log(`FOUND ${shapes.length} MORPH TARGETS`);
//       console.log(shapes);
//     } else {
//       console.error("NO MORPH TARGETS FOUND ON HEAD");
//     }
//   }, [scene]);

  return <primitive object={scene} ref={avatarRef} position={[0, -1.7, 0]} />;
}
