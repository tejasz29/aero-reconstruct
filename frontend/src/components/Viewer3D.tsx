import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { api, type Pose } from "../api";

/** 3D viewer: scene.ply (STEP 10) + camera path (STEP 5), orbit/pan/zoom. */
export default function Viewer3D({ jobId }: { jobId: string }) {
  const mount = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState("loading…");
  const [stats, setStats] = useState({ kept: 0, rejected: 0, nPoints: 0 });

  useEffect(() => {
    if (!jobId || !mount.current) return;
    let dead = false;
    let renderer: THREE.WebGLRenderer | null = null;
    let raf = 0;

    (async () => {
      const el = mount.current!;
      el.innerHTML = "";
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0e14);
      const w = el.clientWidth || 800;
      const h = el.clientHeight || 500;
      const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 1e6);
      camera.position.set(3, -3, 3);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(w, h);
      el.appendChild(renderer.domElement);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(0, 0, 0);

      scene.add(new THREE.AxesHelper(1));
      scene.add(new THREE.AmbientLight(0xffffff, 0.9));

      // Trajectory (STEP 5 camera_poses.csv via /trajectory).
      let poses: Pose[] = [];
      try {
        const t = await api.trajectory(jobId);
        if (dead) return;
        poses = t.poses;
        setStats((s) => ({ ...s, kept: t.kept, rejected: t.rejected }));
        const keptPts = poses.filter((p) => p.kept).map((p) => new THREE.Vector3(...p.t));
        if (keptPts.length > 1) {
          scene.add(
            new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(keptPts),
              new THREE.LineBasicMaterial({ color: 0x7cc4ff }),
            ),
          );
        }
        const dot = new THREE.SphereGeometry(0.03, 8, 8);
        for (const p of poses) {
          const m = new THREE.Mesh(
            dot,
            new THREE.MeshBasicMaterial({ color: p.kept ? 0x4ade80 : 0xf87171 }),
          );
          m.position.set(...p.t);
          scene.add(m);
        }
        if (keptPts.length > 0) {
          const c = new THREE.Box3().setFromPoints(keptPts).getCenter(new THREE.Vector3());
          controls.target.copy(c);
        }
      } catch {
        if (!dead) setInfo("trajectory not ready yet (run reconstruct-poses)");
      }

      // Point cloud (STEP 10 scene.ply via /pointcloud -> scene_url).
      try {
        const pc = await api.pointcloud(jobId);
        if (dead) return;
        setStats((s) => ({ ...s, nPoints: pc.n_points }));
        setInfo(
          `scale ${pc.scale} m/unit · metric-via-scale, NOT absolute CRS · ${pc.n_points} pts`,
        );
        const loader = new PLYLoader();
        loader.load(
          pc.scene_url,
          (geo) => {
            if (dead) return;
            geo.computeVertexNormals();
            const mat = new THREE.PointsMaterial({ size: 0.02, vertexColors: true });
            const pts = new THREE.Points(geo, mat);
            scene.add(pts);
            const box = new THREE.Box3().setFromObject(pts);
            const center = box.getCenter(new THREE.Vector3());
            controls.target.copy(center);
            const size = box.getSize(new THREE.Vector3()).length() || 5;
            camera.position.set(center.x + size, center.y - size, center.z + size);
          },
          undefined,
          () => {
            if (!dead) setInfo("scene.ply unloadable (binary PLY or too large for preview)");
          },
        );
      } catch {
        if (!dead && poses.length === 0) setInfo("no artefacts yet — queue a job first");
        else if (!dead) setInfo("trajectory shown · scene.ply not ready yet");
      }

      const loop = () => {
        if (dead) return;
        raf = requestAnimationFrame(loop);
        controls.update();
        renderer!.render(scene, camera);
      };
      loop();
    })();

    return () => {
      dead = true;
      cancelAnimationFrame(raf);
      renderer?.dispose();
    };
  }, [jobId]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <div className="row" style={{ padding: "8px 12px 0" }}>
        <span style={{ fontSize: 12, color: "#9aa4b8" }}>
          kept {stats.kept} · rejected {stats.rejected} · points {stats.nPoints} · {info}
        </span>
      </div>
      <div ref={mount} style={{ flex: 1, minHeight: 380 }} />
    </div>
  );
}
