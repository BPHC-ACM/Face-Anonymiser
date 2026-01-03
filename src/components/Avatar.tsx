// src/components/Avatar.tsx
import { useAvatarLogic } from "../hooks/useAvatarLogic";

type AvatarProps = {
  avatarUrl: string;
};

export function Avatar({ avatarUrl }: AvatarProps) {
  const { scene, avatarRef } = useAvatarLogic(avatarUrl);

  // If your hook expects the component to render something,
  // you likely return a <primitive /> here. Keeping the
  // existing behavior, just wiring in avatarUrl:
  return (
    scene && (
      <primitive
        ref={avatarRef}
        object={scene}
        position={[0, -2.2, 0]} // whatever you currently use
        scale={[1.3, 1.3, 1.3]}
      />
    )
  );
}
