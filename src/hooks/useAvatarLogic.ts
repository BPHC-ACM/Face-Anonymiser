import { useEffect, useRef } from "react";
import { useFrame, useGraph } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

// Import our new split logic files
import { updateHeadRotation } from "../logic/headRotation";
import { updateFaceExpressions } from "../logic/faceExpressions";

export function useAvatarLogic(avatarUrl: string) {
  const { scene } = useGLTF(avatarUrl);
  const { nodes } = useGraph(scene);
  const avatarRef = useRef<THREE.Group>(null);

  // MediaPipe Refs
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // State for rotation smoothing
  const smoothRotation = useRef(new THREE.Euler());

  // 1. Setup: Hide Body
  useEffect(() => {
    if (!scene) return;
    scene.traverse((child) => {
      const mesh = child as THREE.Mesh; // Type Fix here too
      if (mesh.isMesh) {
        const keepList = [
          "Wolf3D_Head",
          "Wolf3D_Teeth",
          "Wolf3D_Beard",
          "Wolf3D_Glasses",
          "Wolf3D_Headwear",
          "EyeLeft",
          "EyeRight",
        ];
        const name = mesh.name;
        if (!keepList.some((part) => name.includes(part))) {
          mesh.visible = false;
        }
      }
    });
  }, [scene]);

  // 2. Setup: MediaPipe
  useEffect(() => {
    const setupMediaPipe = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU",
          },
          outputFacialTransformationMatrixes: true,
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1,
        }
      );
      const video = document.getElementById("video") as HTMLVideoElement;
      if (video) videoRef.current = video;
    };
    setupMediaPipe();
  }, []);

  // 3. Animation Loop (The Orchestrator)
  useFrame(() => {
    if (
      !faceLandmarkerRef.current ||
      !videoRef.current ||
      videoRef.current.readyState !== 4
    )
      return;

    const result = faceLandmarkerRef.current.detectForVideo(
      videoRef.current,
      Date.now()
    );

    if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
      // A. Call Rotation Logic
      const matrix = result.facialTransformationMatrixes![0];
      if (matrix) {
        updateHeadRotation(nodes, matrix, smoothRotation.current);
      }

      // B. Call Expression Logic
      const blendshapes = result.faceBlendshapes[0].categories;
      if (blendshapes) {
        updateFaceExpressions(scene, blendshapes, nodes);
      }
    }
  });

  return { scene, avatarRef };
}
