import { useEffect, useState } from "react";
import { api, type Job } from "./api";
import AccuracyBanner from "./components/AccuracyBanner";
import ReportsPanel from "./components/ReportsPanel";
import Viewer3D from "./components/Viewer3D";
import "./index.css";

export default function App() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [active, setActive] = useState<string>("");
  const [health, setHealth] = useState<string>("…");

  const refresh = () =>
    api
      .jobs()
      .then((j) => {
        setJobs(j);
        if (!active && j.length > 0) setActive(j[0].id);
      })
      .catch(() => {});

  useEffect(() => {
    api
      .health()
      .then((h) => setHealth(`steps 1–${h.steps_done.length} done · next ${h.next_step}`))
      .catch(() => setHealth("backend offline"));
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <>
      <header>
        <h1>single-pass-3d viewer</h1>
        <span className="sub">{health}</span>
        <span className="sub">
          job{" "}
          <select value={active} onChange={(e) => setActive(e.target.value)}>
            <option value="">—</option>
            {jobs.map((j) => (
              <option key={j.id} value={j.id}>
                {j.id} · {j.status}
              </option>
            ))}
          </select>
        </span>
        <button onClick={refresh}>refresh</button>
      </header>
      <main>
        <aside>
          <AccuracyBanner />
          <h3>Jobs ({jobs.length})</h3>
          <ul>
            {jobs.map((j) => (
              <li key={j.id}>
                <button
                  style={{
                    background: j.id === active ? "#0ea5e9" : "#243044",
                    fontSize: 12,
                  }}
                  onClick={() => setActive(j.id)}
                >
                  {j.id}
                </button>{" "}
                <span style={{ fontSize: 12 }}>{j.status}</span>
                <div style={{ fontSize: 11, color: "#9aa4b8" }}>
                  {j.stages
                    .filter((s) => s.status !== "pending")
                    .map((s) => `${s.name}:${s.status}`)
                    .join(" · ")}
                </div>
              </li>
            ))}
          </ul>
          {jobs.length === 0 && (
            <p style={{ fontSize: 12, color: "#9aa4b8" }}>
              No jobs yet — upload arrives in the next commit; meanwhile the viewer
              falls back to global outputs/ when job id is empty.
            </p>
          )}
          <ReportsPanel jobId={active} />
        </aside>
        <section className="view">
          {active ? (
            <Viewer3D jobId={active} />
          ) : (
            <p style={{ padding: 16, color: "#9aa4b8" }}>
              Select a job to view its point cloud + trajectory.
            </p>
          )}
        </section>
      </main>
    </>
  );
}
