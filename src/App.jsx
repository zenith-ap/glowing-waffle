import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Stars } from "@react-three/drei";
import * as THREE from "three";
import {
  FilesetResolver,
  HandLandmarker,
} from "@mediapipe/tasks-vision";
import "./App.css";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const HALF_SIZE = 2.2;

const INITIAL_CONTROL = {
  x: 0,
  y: 0,
  rotX: 0.25,
  rotY: 0.4,
  rotZ: 0,
  scale: 1,
};

function createParticlePositions() {
  const positions = [];
  const surfaceCount = 14000;
  const innerCount = 4000;

  // Densely distribute particles over all six cube faces.
  for (let i = 0; i < surfaceCount; i++) {
    const face = Math.floor(Math.random() * 6);

    const x = (Math.random() * 2 - 1) * HALF_SIZE;
    const y = (Math.random() * 2 - 1) * HALF_SIZE;
    const z = (Math.random() * 2 - 1) * HALF_SIZE;

    if (face === 0) positions.push(HALF_SIZE, y, z);
    else if (face === 1) positions.push(-HALF_SIZE, y, z);
    else if (face === 2) positions.push(x, HALF_SIZE, z);
    else if (face === 3) positions.push(x, -HALF_SIZE, z);
    else if (face === 4) positions.push(x, y, HALF_SIZE);
    else positions.push(x, y, -HALF_SIZE);
  }

  // Inner particles add depth to the hologram.
  for (let i = 0; i < innerCount; i++) {
    positions.push(
      (Math.random() * 2 - 1) * HALF_SIZE,
      (Math.random() * 2 - 1) * HALF_SIZE,
      (Math.random() * 2 - 1) * HALF_SIZE
    );
  }

  return new Float32Array(positions);
}

function Hologram({ control }) {
  const groupRef = useRef(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.BufferAttribute(createParticlePositions(), 3)
    );
    return geo;
  }, []);

  const edgesGeometry = useMemo(
    () =>
      new THREE.EdgesGeometry(
        new THREE.BoxGeometry(
          HALF_SIZE * 2,
          HALF_SIZE * 2,
          HALF_SIZE * 2
        )
      ),
    []
  );

  const cornerGeometry = useMemo(() => {
    const h = HALF_SIZE;
    const points = [];

    for (const x of [-h, h]) {
      for (const y of [-h, h]) {
        for (const z of [-h, h]) {
          points.push(x, y, z);
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(points, 3)
    );
    return geo;
  }, []);

  useEffect(() => {
    return () => {
      geometry.dispose();
      edgesGeometry.dispose();
      cornerGeometry.dispose();
    };
  }, [geometry, edgesGeometry, cornerGeometry]);

  useFrame((_, delta) => {
    const cube = groupRef.current;
    if (!cube) return;

    cube.position.x = THREE.MathUtils.damp(
      cube.position.x,
      control.x,
      7,
      delta
    );

    cube.position.y = THREE.MathUtils.damp(
      cube.position.y,
      control.y,
      7,
      delta
    );

    cube.rotation.x = THREE.MathUtils.damp(
      cube.rotation.x,
      control.rotX,
      5,
      delta
    );

    cube.rotation.y = THREE.MathUtils.damp(
      cube.rotation.y,
      control.rotY,
      5,
      delta
    );

    cube.rotation.z = THREE.MathUtils.damp(
      cube.rotation.z,
      control.rotZ,
      5,
      delta
    );

    const nextScale = THREE.MathUtils.damp(
      cube.scale.x,
      control.scale,
      6,
      delta
    );

    cube.scale.setScalar(nextScale);
  });

  return (
    <group ref={groupRef}>
      {/* Main dense particle cube */}
      <points geometry={geometry} frustumCulled={false}>
        <pointsMaterial
          color="#45f7ff"
          size={0.018}
          sizeAttenuation
          transparent
          opacity={0.95}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Subtle cube outline */}
      <lineSegments geometry={edgesGeometry}>
        <lineBasicMaterial
          color="#a64dff"
          transparent
          opacity={0.45}
        />
      </lineSegments>

      {/* Brighter corner particles */}
      <points geometry={cornerGeometry}>
        <pointsMaterial
          color="#ffffff"
          size={0.075}
          sizeAttenuation
          transparent
          opacity={1}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </points>
    </group>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const landmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const animationRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastUpdateRef = useRef(0);
  const startingRef = useRef(false);

  const [tracking, setTracking] = useState(false);
  const [handFound, setHandFound] = useState(false);
  const [status, setStatus] = useState("Camera is off");
  const [control, setControl] = useState(INITIAL_CONTROL);

  async function startCamera() {
    if (startingRef.current || tracking) return;
    startingRef.current = true;

    try {
      setStatus("Requesting camera permission...");

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera requires localhost or HTTPS.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      });

      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) throw new Error("Camera preview is unavailable.");

      video.srcObject = stream;
      await video.play();

      setStatus("Loading hand tracking model...");

      const vision = await FilesetResolver.forVisionTasks(WASM_URL);

      let landmarker;

      try {
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });
      } catch (gpuError) {
        console.warn("GPU initialization failed; trying CPU.", gpuError);

        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        });
      }

      landmarkerRef.current = landmarker;
      lastVideoTimeRef.current = -1;
      lastUpdateRef.current = 0;

      setTracking(true);
      setStatus("Show your hand to the camera");

      detectHand();
    } catch (error) {
      console.error("Startup error:", error);
      setStatus(error?.message || "Unable to start the camera.");

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    } finally {
      startingRef.current = false;
    }
  }

  function detectHand() {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !landmarker) return;

    if (
      video.readyState >= 2 &&
      video.currentTime !== lastVideoTimeRef.current
    ) {
      lastVideoTimeRef.current = video.currentTime;

      try {
        const result = landmarker.detectForVideo(
          video,
          performance.now()
        );

        const landmarks = result.landmarks?.[0];

        if (landmarks) {
          setHandFound(true);

          const palmX =
            (landmarks[0].x +
              landmarks[5].x +
              landmarks[9].x +
              landmarks[13].x +
              landmarks[17].x) /
            5;

          const palmY =
            (landmarks[0].y +
              landmarks[5].y +
              landmarks[9].y +
              landmarks[13].y +
              landmarks[17].y) /
            5;

          // Keep the large cube within the visible screen area.
          const x = THREE.MathUtils.clamp(
            (0.5 - palmX) * 6.5,
            -3.2,
            3.2
          );

          const y = THREE.MathUtils.clamp(
            (0.5 - palmY) * 3.2,
            -1.2,
            1.2
          );

          const thumb = landmarks[4];
          const index = landmarks[8];

          const pinchDistance = Math.hypot(
            thumb.x - index.x,
            thumb.y - index.y
          );

          const scale = THREE.MathUtils.clamp(
            0.75 + pinchDistance * 2,
            0.75,
            1.25
          );

          const indexBase = landmarks[5];
          const pinkyBase = landmarks[17];
          const wrist = landmarks[0];

          const tilt = Math.atan2(
            pinkyBase.y - indexBase.y,
            pinkyBase.x - indexBase.x
          );

          const rotX = THREE.MathUtils.clamp(
            (palmY - wrist.y) * 3,
            -1,
            1
          );

          const rotY = (palmX - 0.5) * 2.2;
          const rotZ = -tilt;

          const now = performance.now();

          if (now - lastUpdateRef.current > 30) {
            lastUpdateRef.current = now;

            setControl({
              x,
              y,
              rotX,
              rotY,
              rotZ,
              scale,
            });

            setStatus("Hand detected — move your hand");
          }
        } else {
          setHandFound(false);
          setStatus("No hand detected — show your palm");
        }
      } catch (error) {
        console.error("Detection error:", error);
        setStatus("Tracking error — check the console");
      }
    }

    animationRef.current = requestAnimationFrame(detectHand);
  }

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      landmarkerRef.current?.close();
      landmarkerRef.current = null;

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  return (
    <main className="app">
      <Canvas
        className="scene"
        camera={{ position: [0, 0, 6.5], fov: 55 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: false }}
      >
        <color attach="background" args={["#030610"]} />

        <ambientLight intensity={0.7} />

        <Stars
          radius={80}
          depth={45}
          count={1000}
          factor={2}
          saturation={0}
          fade
          speed={0.4}
        />

        <Hologram control={control} />
      </Canvas>

      <header className="top-bar">
        <div className="brand">
          <span className="brand-dot" />
          <span>
            HOLO<span className="accent">MOTION</span>
          </span>
        </div>

        <div className="live-label">
          <span className={handFound ? "live-dot active" : "live-dot"} />
          {handFound ? "TRACKING ACTIVE" : "WAITING FOR HAND"}
        </div>
      </header>

      <section className="instructions">
        <p className="eyebrow">GESTURE CONTROL SYSTEM</p>
        <h1>Move reality.</h1>
        <p className="subtitle">
          Control a particle hologram with your hand.
        </p>

        <div className="gesture-list">
          <div>
            <span>01</span>
            Move hand
            <small>Move the cube</small>
          </div>

          <div>
            <span>02</span>
            Tilt hand
            <small>Rotate the cube</small>
          </div>

          <div>
            <span>03</span>
            Pinch fingers
            <small>Resize the cube</small>
          </div>
        </div>

        {!tracking && (
          <button className="start-button" onClick={startCamera}>
            <span>▶</span> Start camera
          </button>
        )}

        <p className="status">{status}</p>
      </section>

      <video
        ref={videoRef}
        className="camera-preview"
        autoPlay
        muted
        playsInline
      />

      <div className="camera-caption">
        <span className={handFound ? "camera-dot active" : "camera-dot"} />
        CAMERA FEED
      </div>

      <footer className="bottom-bar">
        <span>3D PARTICLE INTERFACE</span>
        <span>HAND TRACKING · MEDIAPIPE</span>
      </footer>
    </main>
  );
}