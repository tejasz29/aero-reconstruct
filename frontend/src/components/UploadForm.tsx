import { useState } from "react";

/** Upload video (+ optional GPS) and queue Steps 2-10. */
export default function UploadForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [video, setVideo] = useState<File | null>(null);
  const [gps, setGps] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const submit = async () => {
    if (!video) {
      setErr("pick a drone video first (.mp4/.mov/.mkv/.avi)");
      return;
    }
    setBusy(true);
    setErr("");
    try {
      const fd = new FormData();
      fd.append("video", video);
      if (gps) fd.append("gps", gps);
      fd.append("target_fps", "2.0");
      fd.append("max_frames", "200");
      fd.append("stride", "2");
      fd.append("depth_backend", "dummy");
      const r = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const body = (await r.json()) as { id: string };
      onCreated(body.id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h3>New job (Steps 2–10)</h3>
      <div className="row">
        <label style={{ fontSize: 12 }}>
          video{" "}
          <input
            type="file"
            accept=".mp4,.mov,.mkv,.avi"
            onChange={(e) => setVideo(e.target.files?.[0] ?? null)}
          />
        </label>
      </div>
      <div className="row">
        <label style={{ fontSize: 12 }}>
          gps.csv (optional){" "}
          <input type="file" accept=".csv" onChange={(e) => setGps(e.target.files?.[0] ?? null)} />
        </label>
      </div>
      <div className="row">
        <button disabled={busy} onClick={submit}>
          {busy ? "uploading…" : "upload + run pipeline"}
        </button>
      </div>
      {err && <p style={{ color: "#f87171", fontSize: 12 }}>{err}</p>}
      <p style={{ fontSize: 11, color: "#9aa4b8" }}>
        No GPS → convert-gps skips and the cloud stays at scale=1 (relative). A
        rejected alignment is a warning, not a failure.
      </p>
    </div>
  );
}
