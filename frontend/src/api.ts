/* Typed fetch helpers for the FastAPI backend (api/main.py + jobs/artifacts). */

export interface Health {
  ok: boolean;
  steps_done: number[];
  next_step: number;
}

export interface JobStage {
  name: string;
  status: string;
  message: string;
}

export interface Job {
  id: string;
  status: string;
  created_at: string;
  stages: JobStage[];
  warnings: string[];
  error: string;
  outputs: Record<string, unknown>;
}

export interface Pose {
  frame_id: number;
  timestamp_s: number;
  kept: boolean;
  reject_reason: string;
  t: [number, number, number];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

export const api = {
  health: () => req<Health>("/api/health"),
  jobs: () => req<{ jobs: Job[] }>("/api/jobs").then((d) => d.jobs),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  logs: (id: string) => req<{ logs: string[] }>(`/api/jobs/${id}/logs?tail=200`),
  trajectory: (id: string) =>
    req<{ kept: number; rejected: number; poses: Pose[] }>(`/api/jobs/${id}/trajectory`),
  pointcloud: (id: string) =>
    req<{
      scene_url: string;
      n_points: number;
      scale: number;
      absolute_crs: boolean;
      report: Record<string, unknown>;
      fused_url: string | null;
      fused_report: { n_points?: number; kept_ratio?: number } & Record<string, unknown>;
    }>(`/api/jobs/${id}/pointcloud`),
  reports: (id: string) =>
    req<{ reports: Record<string, unknown> }>(`/api/jobs/${id}/reports`),
};
