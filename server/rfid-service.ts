import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import { getRfidServiceBaseUrl } from "./env";

const STARTUP_TIMEOUT_MS = 15000;
const PROBE_TIMEOUT_MS = 1200;
const PROBE_INTERVAL_MS = 400;
const LOCAL_SERVICE_HOSTS = new Set(["127.0.0.1", "localhost", "0.0.0.0", "::1"]);

let managedRfidService: ChildProcessWithoutNullStreams | null = null;
let startPromise: Promise<void> | null = null;
let lastStartupError: string | null = null;

function buildServiceUrl(pathname: string) {
  const baseUrl = getRfidServiceBaseUrl();
  return new URL(pathname, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`).toString();
}

function getConfiguredServiceUrl() {
  return new URL(getRfidServiceBaseUrl());
}

function getManagedServiceConfig() {
  const serviceUrl = getConfiguredServiceUrl();
  return {
    isLocal: LOCAL_SERVICE_HOSTS.has(serviceUrl.hostname),
    host:
      serviceUrl.hostname === "0.0.0.0" || serviceUrl.hostname === "::1"
        ? "127.0.0.1"
        : serviceUrl.hostname || "127.0.0.1",
    port: serviceUrl.port || "8001",
  };
}

function getPythonLaunchCommand() {
  if (process.platform === "win32") {
    return {
      command: "py",
      args: ["-3", "-m", "rfid_service.main"],
    };
  }

  return {
    command: "python3",
    args: ["-m", "rfid_service.main"],
  };
}

function logChunk(level: "info" | "warn", chunk: string) {
  const lines = chunk
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (!lines.length) {
    return;
  }

  if (level === "warn") {
    lastStartupError = lines[lines.length - 1];
  }

  for (const line of lines) {
    console[level](`[rfid-service] ${line}`);
  }
}

function wait(ms: number) {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function isServiceReachable() {
  try {
    const response = await fetch(buildServiceUrl("status"), {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function waitForServiceReady(child: ChildProcessWithoutNullStreams) {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;

  while (Date.now() < deadline) {
    if (await isServiceReachable()) {
      return;
    }

    if (managedRfidService !== child || child.exitCode !== null || child.killed) {
      throw new Error(lastStartupError || "RFID service stopped before it became ready.");
    }

    await wait(PROBE_INTERVAL_MS);
  }

  throw new Error(
    lastStartupError || `RFID service did not become ready within ${STARTUP_TIMEOUT_MS}ms.`,
  );
}

async function startManagedRfidService() {
  if (await isServiceReachable()) {
    return;
  }

  const { isLocal, host, port } = getManagedServiceConfig();
  if (!isLocal) {
    return;
  }

  if (startPromise) {
    await startPromise;
    return;
  }

  startPromise = (async () => {
    lastStartupError = null;
    const { command, args } = getPythonLaunchCommand();
    const child = spawn(command, args, {
      cwd: process.cwd(),
      windowsHide: true,
      env: {
        ...process.env,
        RFID_SERVICE_HOST: host,
        RFID_SERVICE_PORT: port,
      },
    });

    managedRfidService = child;

    child.stdout.on("data", (chunk) => {
      logChunk("info", String(chunk));
    });

    child.stderr.on("data", (chunk) => {
      logChunk("warn", String(chunk));
    });

    child.on("error", (error) => {
      lastStartupError = error.message;
      console.warn("[rfid-service] Failed to start managed service:", error);
    });

    child.on("close", (code) => {
      if (managedRfidService === child) {
        managedRfidService = null;
      }

      if (code !== 0 && code !== null) {
        lastStartupError = lastStartupError || `RFID service stopped with code ${code}.`;
      }
    });

    try {
      await waitForServiceReady(child);
    } catch (error) {
      if (managedRfidService === child && child.exitCode === null && !child.killed) {
        child.kill();
      }
      throw error;
    } finally {
      startPromise = null;
    }
  })();

  await startPromise;
}

export async function ensureRfidServiceAvailable() {
  if (await isServiceReachable()) {
    return true;
  }

  await startManagedRfidService();
  return await isServiceReachable();
}

export async function warmRfidService() {
  await ensureRfidServiceAvailable();
}

export async function stopManagedRfidService() {
  if (!managedRfidService) {
    startPromise = null;
    return;
  }

  const child = managedRfidService;
  managedRfidService = null;
  startPromise = null;

  if (child.exitCode === null && !child.killed) {
    child.kill();
  }
}
