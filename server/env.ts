import fs from "fs";
import path from "path";

const initialEnvKeys = new Set(Object.keys(process.env));
let didLoadEnvironment = false;

function parseEnvValue(rawValue: string) {
  const trimmed = rawValue.trim();

  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"'))
    || (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1).replace(/\\n/g, "\n").replace(/\\r/g, "\r").replace(/\\t/g, "\t");
  }

  return trimmed;
}

function isTruthyEnvValue(value?: string | null) {
  if (!value) {
    return false;
  }

  const normalized = value.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function loadEnvFile(filePath: string) {
  if (!fs.existsSync(filePath)) {
    return;
  }

  const fileContents = fs.readFileSync(filePath, "utf8");

  for (const rawLine of fileContents.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const equalsIndex = line.indexOf("=");
    if (equalsIndex <= 0) {
      continue;
    }

    const key = line.slice(0, equalsIndex).trim();
    const rawValue = line.slice(equalsIndex + 1);

    if (!key || initialEnvKeys.has(key)) {
      continue;
    }

    process.env[key] = parseEnvValue(rawValue);
  }
}

export function loadEnvironment() {
  if (didLoadEnvironment) {
    return;
  }

  const cwd = process.cwd();
  loadEnvFile(path.join(cwd, ".env"));
  loadEnvFile(path.join(cwd, ".env.local"));

  didLoadEnvironment = true;
}

export function getDatabaseUrl() {
  loadEnvironment();

  return process.env.DATABASE_URL ?? process.env.SUPABASE_DB_URL ?? null;
}

export function normalizeConnectionString(databaseUrl: string) {
  try {
    const parsedUrl = new URL(databaseUrl);

    parsedUrl.searchParams.delete("sslmode");
    parsedUrl.searchParams.delete("ssl");
    parsedUrl.searchParams.delete("sslcert");
    parsedUrl.searchParams.delete("sslkey");
    parsedUrl.searchParams.delete("sslrootcert");

    return parsedUrl.toString();
  } catch {
    return databaseUrl;
  }
}

export function shouldEnableSsl(databaseUrl: string) {
  const sslMode = process.env.PGSSLMODE?.toLowerCase();
  if (sslMode && sslMode !== "disable") {
    return true;
  }

  try {
    const parsedUrl = new URL(databaseUrl);
    const urlSslMode = parsedUrl.searchParams.get("sslmode")?.toLowerCase();

    if (urlSslMode && urlSslMode !== "disable") {
      return true;
    }

    return parsedUrl.hostname.includes("supabase");
  } catch {
    return false;
  }
}

export function allowInsecureFaceFallback() {
  loadEnvironment();
  return isTruthyEnvValue(
    process.env.ALLOW_INSECURE_FACE_FALLBACK
    ?? process.env.VITE_ALLOW_INSECURE_FACE_FALLBACK
    ?? null,
  );
}

export function useTriggeredCameraFaceRecognition() {
  loadEnvironment();
  return isTruthyEnvValue(
    process.env.ENABLE_TRIGGERED_CAMERA_FACE_RECOGNITION
    ?? null,
  );
}

export function getRfidServiceBaseUrl() {
  loadEnvironment();
  return process.env.RFID_SERVICE_BASE_URL?.trim() || "http://127.0.0.1:8001";
}
