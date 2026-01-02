import { useAvatarLogic } from "../hooks/useAvatarLogic";

export function Avatar() {
  // Pass the path to your avatar file here
  const { scene, avatarRef } = useAvatarLogic("/avatars/avatar-2.glb");

  // Position adjusted to center the head
  return <primitive object={scene} ref={avatarRef} position={[0, -1.7, 0]} />;
}