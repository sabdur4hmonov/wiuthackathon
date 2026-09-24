import type { RiskPoint } from "./predictions.ts";

export type AdapterAvailability = "checking" | "available" | "unavailable" | "unknown";

export interface AdapterStatusCopy {
  label: string;
  detail: string;
}

export type CoverageStatus =
  | { kind: "unverified" }
  | { kind: "partial"; detail: string }
  | { kind: "verified-full"; detail: string };

export const UNVERIFIED_COVERAGE: CoverageStatus = { kind: "unverified" };

export const PHASE_6_COVERAGE: CoverageStatus = {
  kind: "partial",
  detail: "approximately 0.6 s / 10 processed frames",
};

export const FORMAT_VALIDATION_MESSAGE =
  "Prediction format/schema validated only; detection accuracy was not validated.";

export const EMPTY_EVENT_MESSAGE =
  "No events in the supplied prediction. This does not establish full-video perception coverage or prove that no events occurred.";

export const ALL_ZERO_RISK_MESSAGE =
  "All supplied risk samples are zero. This is recorded model output, not proof that the scene is safe and not an accuracy claim.";

export function adapterStatusCopy(availability: AdapterAvailability): AdapterStatusCopy {
  if (availability === "available") {
    return {
      label: "Model adapter available",
      detail: "The local API reports that a model adapter is connected. Viewing supplied prediction data does not itself run the live model.",
    };
  }
  if (availability === "unavailable") {
    return {
      label: "Model adapter unavailable / disconnected",
      detail: "The local API explicitly reports that no model adapter is connected. Supplied prediction data can still be viewed without live model execution.",
    };
  }
  if (availability === "checking") {
    return {
      label: "Checking model adapter",
      detail: "Waiting for the local API health response; adapter availability has not been inferred.",
    };
  }
  return {
    label: "Model adapter status unavailable",
    detail: "The local API health response could not be confirmed, so adapter availability is unknown rather than inferred.",
  };
}

export function coverageMessage(coverage: CoverageStatus): string {
  if (coverage.kind === "partial") {
    return `Partial coverage only: ${coverage.detail}. Full-video perception coverage is not verified.`;
  }
  if (coverage.kind === "verified-full") {
    return `Full-video perception coverage verified: ${coverage.detail}. This is a coverage statement, not an accuracy claim.`;
  }
  return "Coverage not verified: the supplied prediction does not establish full-video perception coverage.";
}

export function executionMessage(sourceKind: "fixture" | "sample" | "upload"): string {
  if (sourceKind === "upload") {
    return "This prediction was returned by a completed live demo job.";
  }
  return "Viewing supplied prediction data without live model execution.";
}

export function isAllZeroRecordedRisk(risk: RiskPoint[], isRecordedModelOutput: boolean): boolean {
  return isRecordedModelOutput && risk.length > 0 && risk.every(([, score]) => score === 0);
}
