import { useEffect, useState } from "react";
import { api, type Job } from "../api";

/** Polling stage stepper + log tail for one job. */
export default function PipelineProgress({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    if (!jobId) return;
    let dead = false;
    const tick = async () => {
      try {
        const [j, l] = await Promise.all([api.job(jobId), api.logs(jobId)]);
        if (!dead) {
          setJob(j);
          setLogs(l.logs);
        }
      } catch {
        /* backend offline — keep last state */
      }
    };
    tick();
    const t = setInterval(tick, 2000);
    return () => {
      dead = true;
      clearInterval(t);
    };
  }, [jobId]);

  if (!jobId) return null;
  if (!job) return <p>loading job…</p>;
  return (
    <div>
      <h3>
        Pipeline · {job.status} {job.error && <span style={{ color: "#f87171" }}>{job.error}</span>}
      </h3>
      <ul style={{ fontSize: 12, paddingLeft: 18 }}>
        {job.stages.map((s) => (
          <li key={s.name} className={`stage-${s.status}`}>
            {s.name}: {s.status} {s.message}
          </li>
        ))}
      </ul>
      {job.warnings.length > 0 && (
        <details open>
          <summary>warnings ({job.warnings.length})</summary>
          <ul style={{ fontSize: 12 }}>
            {job.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </details>
      )}
      <details>
        <summary>logs ({logs.length})</summary>
        <pre className="logs">{logs.join("\n")}</pre>
      </details>
    </div>
  );
}
