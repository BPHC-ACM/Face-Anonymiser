import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";
import { Avatar } from "./components/Avatar"; // Adjust path if needed

function App() {
  return (
    <div style={{ width: "100vw", height: "100vh", background: "#111" }}>
      <Canvas camera={{ position: [0, 0, 0.8], fov: 50 }}>
        {/* Lights */}
        <ambientLight intensity={0.6} />
        <spotLight
          position={[10, 10, 10]}
          angle={0.15}
          penumbra={1}
          intensity={1}
        />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />

        {/* Render Avatar */}
        <Suspense fallback={null}>
          <Avatar />
        </Suspense>
      </Canvas>
    </div>
  );
}

export default App;
