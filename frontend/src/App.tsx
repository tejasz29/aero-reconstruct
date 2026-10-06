import { useEffect, useState } from "react";
import { api, type Job } from "./api";
import "./index.css";

export default function App() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [health, setHealth] = useState<string>("…");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    api
      .health()
      .then((h) => setHealth(`steps 1–${h.steps_done.length} done · next ${h.next_step}`))
      .catch((e: Error) => setHealth(`backend offline: ${e.message}`));
    api
      .jobs()
      .then(setJobs)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <>
      <header>
        <h1>single-pass-3d viewer</h1>
        <span className="sub">{health}</span>
      </header>
      <main>
        <aside>
          <div className="banner">
            Relative accuracy ≠ absolute accuracy. Ordinary GPS is metre-level;
            clouds are metric-via-GPS-scale until STEP 16 — not absolute CRS.
          </div>
          {error && <p>jobs: {error} (start backend: uvicorn api.main:create_app --factory)</p>}
          <h3>Jobs ({jobs.length})</h3>
          <ul>
            {jobs.map((j) => (
              <li key={j.id}>
                {j.id} — {j.status}
              </li>
            ))}
          </ul>
          <p style={{ color: "#9aa4b8", fontSize: 12 }}>
            Upload + 3D viewer land in the next commits.
          </p>
        </aside>
        <section className="view" id="viewer">
          <p style={{ padding: 16, color: "#9aa4b8" }}>
            Viewer placeholder — Three.js point-cloud + trajectory arrive next.
          </p>
        </section>
      </main>
    </>
  );
}
