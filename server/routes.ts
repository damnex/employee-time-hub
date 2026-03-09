import type { Express } from "express";
import type { Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";

// WebSocket connection manager for ESP8266 devices
const deviceConnections = new Map<string, WebSocket>();

// Helper function to calculate Euclidean distance between two vectors
function euclideanDistance(v1: number[], v2: number[]): number {
  if (!v1 || !v2 || v1.length !== v2.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += Math.pow(v1[i] - v2[i], 2);
  }
  return Math.sqrt(sum);
}

// Face recognition match confidence (1.0 - distance). Usually distance < 0.6 is a match for FaceNet models
function calculateMatchConfidence(distance: number): number {
  return Math.max(0, 1 - distance);
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // Setup WebSocket server for real ESP8266 device communication
  const wss = new WebSocketServer({ server: httpServer, path: "/ws/device" });
  
  wss.on("connection", (ws: WebSocket, req) => {
    const deviceId = new URL(`http://${req.headers.host}${req.url}`).searchParams.get("deviceId") || `device-${Date.now()}`;
    console.log(`[WebSocket] Device connected: ${deviceId}`);
    deviceConnections.set(deviceId, ws);
    
    ws.send(JSON.stringify({ type: "connected", deviceId, message: "Connected to attendance system" }));
    
    ws.on("message", async (data) => {
      try {
        const message = JSON.parse(data.toString());
        
        // Handle RFID scan from ESP8266 device
        if (message.type === "rfid_scan") {
          const { rfidUid, faceDescriptor } = message;
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
            const distance = euclideanDistance(faceDescriptor, employee.faceDescriptor as number[]);
            confidence = calculateMatchConfidence(distance);
            faceMatched = confidence > 0.6;
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
      console.log(`[WebSocket] Device disconnected: ${deviceId}`);
      deviceConnections.delete(deviceId);
    });
  });
  
  // Seed Database on startup if empty
  setTimeout(async () => {
    try {
      const existing = await storage.getEmployees();
      if (existing.length === 0) {
        console.log("Seeding database...");
        
        // Add sample device
        await storage.createDevice({
          deviceId: "gate-1",
          location: "Main Entrance",
          deviceType: "Turnstile"
        });

        // Add sample employees
        const emp1 = await storage.createEmployee({
          employeeCode: "EMP001",
          name: "John Doe",
          department: "Engineering",
          email: "john@example.com",
          phone: "555-0101",
          rfidUid: "A1B2C3D4",
          faceDescriptor: Array.from({length: 128}, () => Math.random()),
          isActive: true
        });

        const emp2 = await storage.createEmployee({
          employeeCode: "EMP002",
          name: "Jane Smith",
          department: "HR",
          email: "jane@example.com",
          phone: "555-0102",
          rfidUid: "E5F6G7H8",
          faceDescriptor: Array.from({length: 128}, () => Math.random()),
          isActive: true
        });
        
        // Add sample attendances
        const today = new Date().toISOString().split('T')[0];
        const entryTime = new Date();
        entryTime.setHours(9, 0, 0, 0);
        
        await storage.createAttendance({
          employeeId: emp1.id,
          date: today,
          entryTime: entryTime,
          verificationStatus: "ENTRY",
          deviceId: "gate-1",
        });

      }
    } catch (e) {
      console.error("Failed to seed database:", e);
    }
  }, 1000);

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
      const employee = await storage.updateEmployee(Number(req.params.id), input);
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

  // Attendances API
  app.get(api.attendances.list.path, async (req, res) => {
    try {
      // @ts-ignore
      const input = api.attendances.list.input ? api.attendances.list.input.parse(req.query) : req.query;
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
        // Compare descriptors
        const distance = euclideanDistance(input.faceDescriptor, employee.faceDescriptor as number[]);
        confidence = calculateMatchConfidence(distance);
        
        // typical threshold for facenet is distance < 0.6, which means confidence > 0.4. Let's use 0.6 as confidence threshold as requested
        faceMatched = confidence > 0.6;
      } else {
        // If the gate didn't provide a face, or employee has no face enrolled, we might allow it or fail it.
        // The requirements say "allow attendance only if match confidence > 0.6".
        // For testing/simulator purposes, if no face was provided by simulator, we simulate success for simplicity? 
        // No, let's strictly enforce face descriptor check.
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
