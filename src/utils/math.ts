import * as THREE from "three";

export const lerp = (start: number, end: number, factor: number) => {
  return start + (end - start) * factor;
};

export const dampEuler = (
  current: THREE.Euler,
  target: THREE.Euler,
  factor: number
) => {
  return new THREE.Euler(
    lerp(current.x, target.x, factor),
    lerp(current.y, target.y, factor),
    lerp(current.z, target.z, factor)
  );
};
