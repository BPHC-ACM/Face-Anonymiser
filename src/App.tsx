import { useEffect } from "react"; // <--- Import useEffect
import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import { Avatar } from "./components/Avatar";

function App() {
  // --- ADD THIS BLOCK TO START THE CAMERA ---
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
  // ------------------------------------------

  return (
    <div
      style={{
        display: "flex",
        width: "100vw",
        height: "100vh",
        background: "#111",
      }}
    >
      {/* Left Side: Video Feed */}
      <div
        style={{
          flex: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          borderRight: "2px solid #333",
          background: "#000",
        }}
      >
        <video
          id="video" // Avatar.tsx will look for this ID later to track your face
          autoPlay
          playsInline
          muted
          style={{
            width: "100%",
            height: "auto",
            transform: "scaleX(-1)", // Mirror effect
          }}
        ></video>
      </div>

      {/* Right Side: Avatar */}
      <div style={{ flex: 1 }}>
        <Canvas camera={{ position: [0, 0, 0.6], fov: 60 }}>
          {" "}
          <ambientLight intensity={1.5} />
          <spotLight
            position={[10, 10, 10]}
            angle={0.15}
            penumbra={1}
            intensity={3}
          />
          <pointLight position={[-10, -10, -10]} intensity={1.5} />
          <Suspense fallback={null}>
            <Avatar />
          </Suspense>
        </Canvas>
      </div>
    </div>
  );
}

export default App;
