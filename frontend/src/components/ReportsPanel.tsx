import { useEffect, useState } from "react";
import { api } from "../api";

/** Raw pipeline reports (STEPS 4-11): calibration, trajectory, GPS, depth, fusion. */
export default function ReportsPanel({ jobId }: { jobId: string }) {
  const [reports, setReports] = useState<Record<string, unknown>>({});
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!jobId) return;
    api
      .reports(jobId)
      .then((d) => {
        setReports(d.reports);
        setErr("");
      })
      .catch((e: Error) => setErr(e.message));
  }, [jobId]);

  if (!jobId) return <p>Select a job to inspect reports.</p>;
  const names = Object.keys(reports);
  return (
    <div>
      <h3>Reports</h3>
      {err && <p>reports: {err}</p>}
      {names.length === 0 && !err && <p>No reports yet for this job.</p>}
      {names.map((n) => (
        <details key={n}>
          <summary>{n}</summary>
          <pre className="logs">{JSON.stringify(reports[n], null, 2)}</pre>
        </details>
      ))}
      <p style={{ fontSize: 11, color: "#9aa4b8" }}>
        depth is relative until STEP 10 applies the STEP 8 GPS scale; RMSE describes
        alignment fit only, not absolute cm-accuracy. STEP 11 fusion only
        averages observed points — gaps stay gaps, nothing is invented.
      </p>
    </div>
  );
}
