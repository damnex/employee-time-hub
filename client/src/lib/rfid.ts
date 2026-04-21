import { apiRequest } from "./queryClient";


export type RfidMode = "normal" | "registration";
export type RfidTransportMode = "scan" | "answer";

export interface RfidReaderInfo {
  version: number;
  reader_type: number;
  protocol_mask: number;
  max_frequency: number;
  min_frequency: number;
  power: number;
  scan_time: number;
}

export interface RfidRegistrationState {
  mode: string;
  selected_tag: string | null;
  candidate_tag: string | null;
  candidate_hits: number;
  stable_threshold: number;
  multiple_tags_detected: boolean;
  message: string;
  selected_at: number | null;
  last_seen_at: number | null;
}

export interface RfidStatus {
  port: string;
  baudrate: number;
  connected: boolean;
  running: boolean;
  current_mode: RfidMode;
  transport_mode: RfidTransportMode;
  buzzer_enabled: boolean;
  current_power: number;
  debug_raw: boolean;
  last_error: string | null;
  reader_info: RfidReaderInfo | null;
}

export interface RfidSerialPortInfo {
  device: string;
  description: string | null;
  manufacturer: string | null;
  hwid: string | null;
  vid: number | null;
  pid: number | null;
}

export interface RfidTagObservation {
  epc: string;
  seen_at: number;
  raw_hex?: string | null;
}

export interface RfidTagsResponse extends RfidStatus {
  last_detected_tag: string | null;
  last_detected_at: number | null;
  last_packet_hex: string | null;
  active_tag_count: number;
  recent_tags: RfidTagObservation[];
  registration: RfidRegistrationState;
}

export interface RfidActiveTag {
  epc: string;
  first_seen_at: number;
  last_seen_at: number;
  detections: number;
  age_seconds: number;
  idle_seconds: number;
}

export interface RfidActiveTagsResponse extends RfidStatus {
  active_tags: RfidActiveTag[];
}

export interface RfidRegistrationResponse extends RfidStatus {
  selected_tag: string | null;
  registration: RfidRegistrationState;
}

export interface RfidDetectPortResponse extends RfidStatus {
  detected_port: RfidSerialPortInfo;
  detected_reader_info: RfidReaderInfo | null;
}

export const rfidQueryKeys = {
  connection: ["/api/rfid/status"] as const,
  tags: ["/api/rfid/tags"] as const,
  activeTags: ["/api/rfid/active-tags"] as const,
  registrationTag: ["/api/rfid/registration-tag"] as const,
  status: ["/api/rfid/status"] as const,
};

async function readJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    credentials: "include",
    ...init,
  });

  if (!response.ok) {
    const text = (await response.text()) || response.statusText;
    throw new Error(text);
  }

  return response.json() as Promise<T>;
}

export function fetchRfidTags() {
  return readJson<RfidTagsResponse>("/api/rfid/tags");
}

export function fetchRfidStatus() {
  return readJson<RfidStatus>("/api/rfid/status");
}

export function fetchRfidActiveTags() {
  return readJson<RfidActiveTagsResponse>("/api/rfid/active-tags");
}

export function fetchRfidRegistrationTag() {
  return readJson<RfidRegistrationResponse>("/api/rfid/registration-tag");
}

export async function connectRfidReader(payload: {
  port: string;
  baudrate: number;
  debug_raw?: boolean;
}) {
  const response = await apiRequest("POST", "/api/rfid/connect", payload);
  return response.json() as Promise<RfidStatus>;
}

export async function disconnectRfidReader() {
  const response = await apiRequest("POST", "/api/rfid/disconnect");
  return response.json() as Promise<RfidStatus>;
}

export async function detectRfidPort(payload?: {
  baudrate?: number;
  debug_raw?: boolean;
}) {
  const response = await apiRequest("POST", "/api/rfid/detect-port", payload);
  return response.json() as Promise<RfidDetectPortResponse>;
}

export async function startRfidReader(payload?: {
  port?: string;
  baudrate?: number;
  debug_raw?: boolean;
}) {
  const response = await apiRequest("POST", "/api/rfid/start", payload);
  return response.json() as Promise<RfidStatus>;
}

export async function stopRfidReader() {
  const response = await apiRequest("POST", "/api/rfid/stop");
  return response.json() as Promise<RfidStatus>;
}

export async function setRfidPower(level: number) {
  const response = await apiRequest("POST", "/api/rfid/set-power", { level });
  return response.json() as Promise<RfidStatus>;
}

export async function setRfidMode(mode: RfidMode) {
  const response = await apiRequest("POST", "/api/rfid/set-mode", { mode });
  return response.json() as Promise<RfidStatus>;
}

export async function setRfidTransportMode(mode: RfidTransportMode) {
  const response = await apiRequest("POST", "/api/rfid/set-transport-mode", { mode });
  return response.json() as Promise<RfidStatus>;
}

export async function setRfidBuzzer(enabled: boolean) {
  const response = await apiRequest("POST", "/api/rfid/set-buzzer", { enabled });
  return response.json() as Promise<RfidStatus>;
}
