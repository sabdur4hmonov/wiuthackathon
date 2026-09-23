import type { JobView } from "./api.ts";
import type { ParsedPredictions } from "./predictions.ts";

/** Selection and submitted video are deliberately separate. */
export interface UploadRun {
  id: number;
  file: File;
  job: JobView | null;
  result: ParsedPredictions | null;
  durationSec?: number;
  uploading: boolean;
  error: string | null;
}

export interface UploadState {
  selectedFile: File | null;
  run: UploadRun | null;
}

export const initialUploadState: UploadState = { selectedFile: null, run: null };

export type UploadAction =
  | { type: "select"; file: File | null }
  | { type: "start"; id: number; file: File }
  | { type: "created"; id: number; job: JobView }
  | { type: "status"; id: number; job: JobView }
  | { type: "result"; id: number; jobId: string; result: ParsedPredictions }
  | { type: "duration"; id: number; seconds: number }
  | { type: "uploadDone"; id: number }
  | { type: "error"; id: number; message: string };

export function uploadReducer(state: UploadState, action: UploadAction): UploadState {
  if (action.type === "select") return { ...state, selectedFile: action.file };
  if (action.type === "start") {
    if (state.run && action.id <= state.run.id) return state;
    return {
      ...state,
      run: { id: action.id, file: action.file, job: null, result: null, uploading: true, error: null },
    };
  }

  const run = state.run;
  if (!run || action.id !== run.id) return state;
  switch (action.type) {
    case "created":
      if (action.job.filename !== run.file.name) {
        return { ...state, run: { ...run, error: "Upload job filename mismatch." } };
      }
      return { ...state, run: { ...run, job: action.job } };
    case "status":
      if (action.job.job_id !== run.job?.job_id || action.job.filename !== run.file.name) return state;
      return { ...state, run: { ...run, job: action.job, error: null } };
    case "result":
      if (action.jobId !== run.job?.job_id ||
          Object.keys(action.result.document.videos).length !== 1 ||
          !Object.hasOwn(action.result.document.videos, run.file.name)) return state;
      return { ...state, run: { ...run, result: action.result, error: null } };
    case "duration":
      if (!Number.isFinite(action.seconds) || action.seconds <= 0) return state;
      return { ...state, run: { ...run, durationSec: action.seconds } };
    case "uploadDone":
      return { ...state, run: { ...run, uploading: false } };
    case "error":
      return { ...state, run: { ...run, error: action.message, uploading: false } };
  }
}
