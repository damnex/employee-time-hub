import type { Express } from "express";
import type { Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";

type ClientType = "browser" | "device";

interface ClientConnection {
  ws: WebSocket;
  deviceId: string;
  clientType: ClientType;
}

const activeConnections = new Set<ClientConnection>();

function normalizeDescriptor(descriptor: number[]) {
  if (!descriptor.length) {
    return [];
  }

  const mean =
    descriptor.reduce((sum, value) => sum + value, 0) / descriptor.length;
  const centered = descriptor.map((value) => value - mean);
  const magnitude = Math.sqrt(
    centered.reduce((sum, value) => sum + value * value, 0),
  );

  if (!magnitude) {
    return centered.map(() => 0);
  }

  return centered.map((value) => value / magnitude);
}

function calculateMatchConfidence(v1: number[], v2: number[]): number {
  if (!v1 || !v2 || v1.length !== v2.length || !v1.length) {
    return 0;
  }

  const normalizedV1 = normalizeDescriptor(v1);
  const normalizedV2 = normalizeDescriptor(v2);

  let dotProduct = 0;
  for (let i = 0; i < normalizedV1.length; i++) {
    dotProduct += normalizedV1[i] * normalizedV2[i];
  }

  return Number((((dotProduct + 1) / 2)).toFixed(4));
}

const FACE_MATCH_THRESHOLD = 0.72;

function broadcastMessage(
  payload: Record<string, unknown>,
  predicate?: (connection: ClientConnection) => boolean,
) {
  const message = JSON.stringify(payload);

  activeConnections.forEach((connection) => {
    if (connection.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    if (predicate && !predicate(connection)) {
      return;
    }

    connection.ws.send(message);
  });
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // Setup WebSocket server for real ESP8266 device communication
  const wss = new WebSocketServer({ server: httpServer, path: "/ws/device" });
  
  wss.on("connection", (ws: WebSocket, req) => {
    const searchParams = new URL(
      `http://${req.headers.host}${req.url}`,
    ).searchParams;
    const deviceId =
      searchParams.get("deviceId") || `device-${Date.now()}`;
    const clientType = (searchParams.get("clientType") === "device"
      ? "device"
      : "browser") as ClientType;
    const connection: ClientConnection = { ws, deviceId, clientType };

    console.log(`[WebSocket] ${clientType} connected: ${deviceId}`);
    activeConnections.add(connection);
    
    ws.send(JSON.stringify({
      type: "connected",
      deviceId,
      message: "Connected to attendance system",
    }));
    
    ws.on("message", async (data) => {
      try {
        const message = JSON.parse(data.toString());

        if (message.type === "rfid_detected") {
          const rfidUid = String(message.rfidUid || "").trim().toUpperCase();

          if (!rfidUid) {
            ws.send(JSON.stringify({
              type: "error",
              message: "RFID UID is required.",
            }));
            return;
          }

          const mappedEmployee = await storage.getEmployeeByRfid(rfidUid);
          const payload = {
            type: "rfid_detected",
            message: mappedEmployee
              ? `Badge already mapped to ${mappedEmployee.name}.`
              : "Badge detected and ready to assign.",
            rfidUid,
            available: !mappedEmployee,
            employee: mappedEmployee
              ? { id: mappedEmployee.id, name: mappedEmployee.name }
              : undefined,
            deviceId,
          };

          ws.send(JSON.stringify(payload));
          broadcastMessage(payload, (client) => client.clientType === "browser");
          return;
        }
        
        // Handle RFID scan from ESP8266 device
        if (message.type === "rfid_scan") {
          const rfidUid = String(message.rfidUid || "").trim().toUpperCase();
          const { faceDescriptor } = message;
          const now = new Date();
          const todayDate = now.toISOString().split('T')[0];
          
          // Identify employee by RFID
          const employee = await storage.getEmployeeByRfid(rfidUid);
          
          if (!employee) {
            ws.send(JSON.stringify({
              type: "scan_result",
              success: false,
              message: "Unknown RFID card rejected.",
              rfidUid
            }));
            return;
          }
          
          // Face verification
          let confidence = 0;
          let faceMatched = false;
          
          if (faceDescriptor && employee.faceDescriptor) {
            confidence = calculateMatchConfidence(
              faceDescriptor,
              employee.faceDescriptor as number[],
            );
            faceMatched = confidence >= FACE_MATCH_THRESHOLD;
          } else {
            faceMatched = false;
          }
          
          if (!faceMatched) {
            const attendance = await storage.createAttendance({
              employeeId: employee.id,
              date: todayDate,
              entryTime: now,
              verificationStatus: "FAILED_FACE",
              deviceId
            });
            
            ws.send(JSON.stringify({
              type: "scan_result",
              success: false,
              message: "Face verification failed. Access denied.",
              employee: { id: employee.id, name: employee.name },
              matchConfidence: confidence,
              rfidUid
            }));
            return;
          }
          
          // Mark Entry or Exit
          const openEntry = await storage.getOpenAttendance(employee.id, todayDate);
          
          if (openEntry) {
            // Mark EXIT
            const workingHoursMs = now.getTime() - (openEntry.entryTime?.getTime() || now.getTime());
            const workingHours = workingHoursMs / (1000 * 60 * 60);
            
            const attendance = await storage.updateAttendance(openEntry.id, {
              exitTime: now,
              workingHours: Number(workingHours.toFixed(2)),
              verificationStatus: "EXIT"
            });
            
            ws.send(JSON.stringify({
              type: "scan_result",
              success: true,
              message: "Exit marked successfully.",
              action: "EXIT",
              employee: { id: employee.id, name: employee.name },
              matchConfidence: confidence,
              rfidUid
            }));
          } else {
            // Mark ENTRY
            const attendance = await storage.createAttendance({
              employeeId: employee.id,
              date: todayDate,
              entryTime: now,
              verificationStatus: "ENTRY",
              deviceId
            });
            
            ws.send(JSON.stringify({
              type: "scan_result",
              success: true,
              message: "Entry marked successfully.",
              action: "ENTRY",
              employee: { id: employee.id, name: employee.name },
              matchConfidence: confidence,
              rfidUid
            }));
          }
        }
      } catch (error) {
        console.error("[WebSocket] Error processing message:", error);
        ws.send(JSON.stringify({ type: "error", message: "Error processing request" }));
      }
    });
    
    ws.on("close", () => {
      console.log(`[WebSocket] ${clientType} disconnected: ${deviceId}`);
      activeConnections.delete(connection);
    });
  });
  
  async function validateEmployeeIdentityConflicts(
    input: { employeeCode?: string; rfidUid?: string },
    currentEmployeeId?: number,
  ) {
    const employees = await storage.getEmployees();

    if (input.employeeCode) {
      const existingByCode = employees.find((employee) => {
        return employee.employeeCode === input.employeeCode
          && employee.id !== currentEmployeeId;
      });

      if (existingByCode) {
        return {
          field: "employeeCode",
          message: `Employee code already belongs to ${existingByCode.name}.`,
        };
      }
    }

    if (input.rfidUid) {
      const normalizedRfidUid = input.rfidUid.trim().toUpperCase();
      const existingByRfid = employees.find((employee) => {
        return employee.rfidUid.toUpperCase() === normalizedRfidUid
          && employee.id !== currentEmployeeId;
      });

      if (existingByRfid) {
        return {
          field: "rfidUid",
          message: `RFID badge already mapped to ${existingByRfid.name}.`,
        };
      }
    }

    return null;
  }

  // Employees API
  app.get(api.employees.list.path, async (req, res) => {
    const employees = await storage.getEmployees();
    res.json(employees);
  });

  app.get(api.employees.get.path, async (req, res) => {
    const employee = await storage.getEmployee(Number(req.params.id));
    if (!employee) {
      return res.status(404).json({ message: "Employee not found" });
    }
    res.json(employee);
  });

  app.post(api.employees.create.path, async (req, res) => {
    try {
      const input = api.employees.create.input.parse(req.body);
      input.rfidUid = input.rfidUid.trim().toUpperCase();

      const conflict = await validateEmployeeIdentityConflicts(input);
      if (conflict) {
        return res.status(400).json(conflict);
      }

      const employee = await storage.createEmployee(input);
      res.status(201).json(employee);
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }
      res.status(500).json({ message: "Internal server error" });
    }
  });

  app.patch(api.employees.update.path, async (req, res) => {
    try {
      const input = api.employees.update.input.parse(req.body);
      if (input.rfidUid) {
        input.rfidUid = input.rfidUid.trim().toUpperCase();
      }

      const employeeId = Number(req.params.id);
      const conflict = await validateEmployeeIdentityConflicts(input, employeeId);
      if (conflict) {
        return res.status(400).json(conflict);
      }

      const employee = await storage.updateEmployee(employeeId, input);
      if (!employee) {
        return res.status(404).json({ message: "Employee not found" });
      }
      res.json(employee);
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }
      res.status(500).json({ message: "Internal server error" });
    }
  });

  app.delete(api.employees.delete.path, async (req, res) => {
    try {
      const employeeId = Number(req.params.id);
      const deletedEmployee = await storage.deleteEmployee(employeeId);

      if (!deletedEmployee) {
        return res.status(404).json({ message: "Employee not found" });
      }

      res.json(deletedEmployee);
    } catch (err) {
      res.status(500).json({ message: "Internal server error" });
    }
  });

  // Attendances API
  app.get(api.attendances.list.path, async (req, res) => {
    try {
      const parsedInput = api.attendances.list.input?.safeParse(req.query);
      const input = parsedInput?.success ? parsedInput.data : undefined;
      const attendances = await storage.getAttendances(input?.date, input?.employeeId);
      res.json(attendances);
    } catch (err) {
      res.json(await storage.getAttendances());
    }
  });

  // Stats API
  app.get(api.stats.dashboard.path, async (req, res) => {
    const stats = await storage.getDashboardStats();
    res.json(stats);
  });

  // RFID Scan Endpoint (Core Logic)
  app.post(api.scan.rfid.path, async (req, res) => {
    try {
      const input = api.scan.rfid.input.parse(req.body);
      input.rfidUid = input.rfidUid.trim().toUpperCase();
      const now = new Date();
      const todayDate = now.toISOString().split('T')[0];

      // 1. Identify employee by RFID
      const employee = await storage.getEmployeeByRfid(input.rfidUid);
      
      if (!employee) {
        // Unknown RFID
        const failRecord = await storage.createAttendance({
          // Provide a dummy ID or make employeeId nullable in reality, but since our schema requires employeeId,
          // If we want to record failed attempts without employee, we'd need employeeId to be nullable.
          // For now, since schema requires employeeId, we will skip saving unknown RFID completely or we return early.
          // In a real prod environment we'd have a separate table for generic failed access logs.
          employeeId: 0, // This will fail foreign key constraint if we try to insert. Let's just return error.
          date: todayDate,
          entryTime: now,
          verificationStatus: "UNKNOWN_RFID",
          deviceId: input.deviceId
        }).catch(() => null);

        return res.json({
          success: false,
          message: "Unknown RFID card rejected.",
        });
      }

      // 2. Face Verification
      let confidence = 0;
      let faceMatched = false;

      if (input.faceDescriptor && employee.faceDescriptor) {
        confidence = calculateMatchConfidence(
          input.faceDescriptor,
          employee.faceDescriptor as number[],
        );
        faceMatched = confidence >= FACE_MATCH_THRESHOLD;
      } else {
        faceMatched = false;
      }

      if (!faceMatched) {
        const attendance = await storage.createAttendance({
          employeeId: employee.id,
          date: todayDate,
          entryTime: now,
          verificationStatus: "FAILED_FACE",
          deviceId: input.deviceId
        });

        return res.json({
          success: false,
          message: "Face verification failed. Access denied.",
          employee,
          attendance,
          matchConfidence: confidence
        });
      }

      // 3. Mark Entry or Exit automatically
      const openEntry = await storage.getOpenAttendance(employee.id, todayDate);

      if (openEntry) {
        // Mark EXIT
        const workingHoursMs = now.getTime() - (openEntry.entryTime?.getTime() || now.getTime());
        const workingHours = workingHoursMs / (1000 * 60 * 60); // Convert to hours

        const attendance = await storage.updateAttendance(openEntry.id, {
          exitTime: now,
          workingHours: Number(workingHours.toFixed(2)),
          verificationStatus: "EXIT"
        });

        return res.json({
          success: true,
          message: "Exit marked successfully.",
          employee,
          attendance,
          matchConfidence: confidence
        });
      } else {
        // Mark ENTRY
        const attendance = await storage.createAttendance({
          employeeId: employee.id,
          date: todayDate,
          entryTime: now,
          verificationStatus: "ENTRY",
          deviceId: input.deviceId
        });

        return res.json({
          success: true,
          message: "Entry marked successfully.",
          employee,
          attendance,
          matchConfidence: confidence
        });
      }

    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }
      console.error("Scan error:", err);
      res.status(500).json({ message: "Internal server error" });
    }
  });

  return httpServer;
}
