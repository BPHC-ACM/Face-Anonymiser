// src/App.tsx
import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import { Avatar } from "./components/Avatar";

function App() {
  const [selectedAvatar, setSelectedAvatar] = useState("/avatars/avatar-1.glb");

  useEffect(() => {
    const startWebcam = async () => {
      try {
        const video = document.getElementById("video") as HTMLVideoElement;
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
        });
        video.srcObject = stream;
        video.play();
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };
    startWebcam();
  }, []);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        height: "100vh",
      }}
    >
      {/* Left side: webcam */}
      <div style={{ background: "black" }}>
        <video
          id="video"
          style={{ width: "100%", height: "100%", objectFit: "cover",transform: "scaleX(-1)" }}
          autoPlay
          muted
        />
      </div>

      {/* Right side: avatar + selector */}
      <div style={{ background: "black", position: "relative" }}>
        {/* Avatar selector UI */}
        <div
          style={{
            position: "absolute",
            top: 16,
            left: 16,
            zIndex: 10,
            display: "flex",
            gap: 8,
          }}
        >
          <button
            onClick={() => setSelectedAvatar("/avatars/avatar-1.glb")}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "none",
              cursor: "pointer",
              background:
                selectedAvatar === "/avatars/avatar-1.glb"
                  ? "#ffffff"
                  : "#555555",
              color:
                selectedAvatar === "/avatars/avatar-1.glb"
                  ? "#000000"
                  : "#ffffff",
            }}
          >
            Avatar 1
          </button>

          <button
            onClick={() => setSelectedAvatar("/avatars/avatar-2.glb")}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "none",
              cursor: "pointer",
              background:
                selectedAvatar === "/avatars/avatar-2.glb"
                  ? "#ffffff"
                  : "#555555",
              color:
                selectedAvatar === "/avatars/avatar-2.glb"
                  ? "#000000"
                  : "#ffffff",
            }}
          >
            Avatar 2
          </button>

          <button
            onClick={() => setSelectedAvatar("/avatars/avatar-3.glb")}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "none",
              cursor: "pointer",
              background:
                selectedAvatar === "/avatars/avatar-3.glb"
                  ? "#ffffff"
                  : "#555555",
              color:
                selectedAvatar === "/avatars/avatar-3.glb"
                  ? "#000000"
                  : "#ffffff",
            }}
          >
            Avatar 3
          </button>

            <button
            onClick={() => setSelectedAvatar("/avatars/avatar-4.glb")}
            style={{
              padding: "6px 10px",
              borderRadius: 4,
              border: "none",
              cursor: "pointer",
              background:
                selectedAvatar === "/avatars/avatar-3.glb"
                  ? "#ffffff"
                  : "#555555",
              color:
                selectedAvatar === "/avatars/avatar-3.glb"
                  ? "#000000"
                  : "#ffffff",
            }}
          >
            Avatar 4
          </button>

        </div>

        {/* 3D canvas */}
        <Canvas camera={{ position: [0, 0, 1.8], fov: 30 }}>
          <Suspense fallback={null}>
            <ambientLight intensity={0.7} />
            <directionalLight position={[1, 1, 1]} intensity={0.7} />
            <Avatar avatarUrl={selectedAvatar} />
          </Suspense>
        </Canvas>
      </div>
    </div>
  );
}

export default App;
