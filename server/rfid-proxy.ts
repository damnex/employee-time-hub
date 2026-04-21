import type { Express, Request, Response } from "express";

import { getRfidServiceBaseUrl } from "./env";
import { ensureRfidServiceAvailable } from "./rfid-service";


function buildServiceUrl(path: string) {
  const baseUrl = getRfidServiceBaseUrl();
  return new URL(path, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`).toString();
}

function sendProxyError(res: Response, error: unknown) {
  const message = error instanceof Error ? error.message : "RFID service unavailable";
  return res.status(502).json({
    message: "RFID service unavailable.",
    detail: message,
  });
}

async function proxyJsonRequest(
  res: Response,
  options: {
    method: string;
    path: string;
    body?: unknown;
  },
) {
  try {
    await ensureRfidServiceAvailable();

    const response = await fetch(buildServiceUrl(options.path), {
      method: options.method,
      headers: options.body ? { "Content-Type": "application/json" } : {},
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: AbortSignal.timeout(5000),
    });

    const text = await response.text();
    const hasJson = response.headers.get("content-type")?.includes("application/json");
    const payload = hasJson && text ? JSON.parse(text) : text;
    return res.status(response.status).json(payload);
  } catch (error) {
    return sendProxyError(res, error);
  }
}

export function registerRfidProxyRoutes(app: Express) {
  const connectHandler = async (req: Request, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "connect",
      body: req.body ?? {},
    });
  };

  const disconnectHandler = async (_req: unknown, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "disconnect",
    });
  };

  const detectPortHandler = async (req: Request, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "detect-port",
      body: req.body ?? {},
    });
  };

  const startHandler = async (req: Request, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "start",
      body: req.body ?? {},
    });
  };

  const stopHandler = async (_req: unknown, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "stop",
    });
  };

  const setPowerHandler = async (req: Request, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "set-power",
      body: req.body ?? {},
    });
  };

  const setModeHandler = async (req: Request, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "set-mode",
      body: req.body ?? {},
    });
  };

  const setTransportModeHandler = async (req: Request, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "set-transport-mode",
      body: req.body ?? {},
    });
  };

  const setBuzzerHandler = async (req: Request, res: Response) => {
    return proxyJsonRequest(res, {
      method: "POST",
      path: "set-buzzer",
      body: req.body ?? {},
    });
  };

  const tagsHandler = async (_req: unknown, res: Response) => {
    return proxyJsonRequest(res, {
      method: "GET",
      path: "tags",
    });
  };

  const activeTagsHandler = async (_req: unknown, res: Response) => {
    return proxyJsonRequest(res, {
      method: "GET",
      path: "active-tags",
    });
  };

  app.post("/api/rfid/connect", connectHandler);
  app.post("/api/connect", connectHandler);

  app.post("/api/rfid/disconnect", disconnectHandler);
  app.post("/api/disconnect", disconnectHandler);

  app.post("/api/rfid/detect-port", detectPortHandler);
  app.post("/api/detect-port", detectPortHandler);

  app.post("/api/rfid/start", startHandler);
  app.post("/api/start", startHandler);

  app.post("/api/rfid/stop", stopHandler);
  app.post("/api/stop", stopHandler);

  app.post("/api/rfid/set-power", setPowerHandler);
  app.post("/api/set-power", setPowerHandler);

  app.post("/api/rfid/set-mode", setModeHandler);
  app.post("/api/set-mode", setModeHandler);

  app.post("/api/rfid/set-transport-mode", setTransportModeHandler);
  app.post("/api/set-transport-mode", setTransportModeHandler);

  app.post("/api/rfid/set-buzzer", setBuzzerHandler);
  app.post("/api/set-buzzer", setBuzzerHandler);

  app.get("/api/rfid/tags", tagsHandler);
  app.get("/api/tags", tagsHandler);

  app.get("/api/rfid/active-tags", activeTagsHandler);
  app.get("/api/active-tags", activeTagsHandler);

  app.get("/api/rfid/registration-tag", async (_req, res) => {
    return proxyJsonRequest(res, {
      method: "GET",
      path: "registration-tag",
    });
  });
  app.get("/api/registration-tag", async (_req, res) => {
    return proxyJsonRequest(res, {
      method: "GET",
      path: "registration-tag",
    });
  });

  app.get("/api/rfid/status", async (_req, res) => {
    return proxyJsonRequest(res, {
      method: "GET",
      path: "status",
    });
  });
  app.get("/api/status", async (_req, res) => {
    return proxyJsonRequest(res, {
      method: "GET",
      path: "status",
    });
  });
}
