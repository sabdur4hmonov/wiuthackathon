export type JobStatus = "queued" | "running" | "awaiting_model" | "completed" | "failed";

export interface JobView {
  job_id: string;
  status: JobStatus;
  filename: string;
  result_available: boolean;
  message: string;
}

async function readJson(response: Response): Promise<unknown> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`Demo API returned an unreadable response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const error = payload && typeof payload === "object" ? payload as { error?: string; message?: string } : {};
    throw new Error(error.message ?? error.error ?? `Demo API returned HTTP ${response.status}.`);
  }
  return payload;
}

export async function createJob(file: File, signal?: AbortSignal): Promise<JobView> {
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: {
      "Content-Type": "video/mp4",
      "X-File-Name": encodeURIComponent(file.name),
    },
    body: file,
    signal,
  });
  return (await readJson(response)) as JobView;
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<JobView> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
  return (await readJson(response)) as JobView;
}

export async function getJobResult(jobId: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/result`, { signal });
  return readJson(response);
}
