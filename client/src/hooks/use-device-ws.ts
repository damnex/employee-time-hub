import { useEffect, useRef, useState, useCallback } from 'react';

const DEVICE_ONLINE_TTL_MS = 9000;
const DEVICE_ONLINE_SWEEP_MS = 3000;

export interface DeviceScanResult {
  type: 'scan_result' | 'connected' | 'error' | 'rfid_detected' | 'device_presence';
  success?: boolean;
  ignored?: boolean;
  message: string;
  employee?: { id: number; name: string };
  action?: 'ENTRY' | 'EXIT';
  matchConfidence?: number;
  matchDetails?: {
    primaryConfidence: number;
    anchorAverage: number;
    peakAnchorConfidence: number;
    strongAnchorRatio: number;
    liveConsistency: number;
    poseConfidence?: number;
    liveLiveness?: number;
    liveRealness?: number;
  };
  movementDirection?: 'ENTRY' | 'EXIT' | 'UNKNOWN';
  movementConfidence?: number;
  detectedFaceLabel?: string;
  detectedFaceBox?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  } | null;
  rfidUid?: string;
  available?: boolean;
  deviceId?: string;
  online?: boolean;
}

interface UseDeviceWSOptions {
  clientType?: 'browser' | 'device';
}

export function useDeviceWS(
  deviceId?: string,
  options: UseDeviceWSOptions = {},
) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const shouldReconnectRef = useRef(true);
  const generatedDeviceIdRef = useRef(`web-${Math.random().toString(36).slice(2, 10)}`);
  const [isConnected, setIsConnected] = useState(false);
  const [deviceOnline, setDeviceOnline] = useState(false); // presence of any hardware device
  const [lastScanResult, setLastScanResult] = useState<DeviceScanResult | null>(null);
  const clientType = options.clientType ?? 'browser';
  const resolvedDeviceId = deviceId ?? generatedDeviceIdRef.current;
  const onlineDevicesRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/device?deviceId=${encodeURIComponent(resolvedDeviceId)}&clientType=${encodeURIComponent(clientType)}`;
    shouldReconnectRef.current = true;

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const pruneStaleDevices = () => {
      const now = Date.now();
      Array.from(onlineDevicesRef.current.entries()).forEach(([targetDeviceId, lastSeenAt]) => {
        if (now - lastSeenAt > DEVICE_ONLINE_TTL_MS) {
          onlineDevicesRef.current.delete(targetDeviceId);
        }
      });

      setDeviceOnline(onlineDevicesRef.current.size > 0);
    };

    const setKnownDeviceOnline = (targetDeviceId: string, online: boolean) => {
      if (!targetDeviceId || targetDeviceId === resolvedDeviceId) {
        return;
      }

      if (online) {
        onlineDevicesRef.current.set(targetDeviceId, Date.now());
      } else {
        onlineDevicesRef.current.delete(targetDeviceId);
      }

      setDeviceOnline(onlineDevicesRef.current.size > 0);
    };

    const scheduleReconnect = () => {
      if (!shouldReconnectRef.current) {
        return;
      }

      clearReconnectTimer();
      const retryDelayMs = Math.min(2000, 250 * Math.max(1, reconnectAttemptRef.current));
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        connect();
      }, retryDelayMs);
    };

    const connect = () => {
      if (!shouldReconnectRef.current) {
        return;
      }

      if (
        wsRef.current
        && (
          wsRef.current.readyState === WebSocket.OPEN
          || wsRef.current.readyState === WebSocket.CONNECTING
        )
      ) {
        return;
      }

      try {
        const socket = new WebSocket(wsUrl);
        socket.binaryType = 'arraybuffer';
        wsRef.current = socket;

        socket.onopen = () => {
          console.log('[DeviceWS] Connected to device server:', wsUrl);
          reconnectAttemptRef.current = 0;
          setIsConnected(true);
          pruneStaleDevices();
        };

        socket.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as DeviceScanResult;
            setLastScanResult(message);
            if (message.type === "device_presence") {
              const online = Boolean((message as any).online);
              const id = (message as any).deviceId || "unknown";
              setKnownDeviceOnline(id, online);
            } else if (
              clientType === "browser"
              && message.deviceId
              && (message.type === "rfid_detected" || message.type === "scan_result")
            ) {
              // Any real-time device event proves the reader is online even if the
              // presence event arrived late or was missed during a reconnect.
              setKnownDeviceOnline(message.deviceId, true);
            }
            console.log('[DeviceWS] Received:', message);
          } catch (error) {
            console.error('[DeviceWS] Error parsing message:', error);
          }
        };

        socket.onerror = (error) => {
          console.error('[DeviceWS] Error:', error);
        };

        socket.onclose = (event) => {
          if (wsRef.current === socket) {
            wsRef.current = null;
          }

          setIsConnected(false);
          onlineDevicesRef.current.clear();
          setDeviceOnline(false);
          reconnectAttemptRef.current += 1;
          console.log(
            `[DeviceWS] Disconnected (code=${event.code}). Reconnecting attempt ${reconnectAttemptRef.current}...`,
          );
          scheduleReconnect();
        };
      } catch (error) {
        console.error('[DeviceWS] Failed to create WebSocket:', error);
        setIsConnected(false);
        setDeviceOnline(false);
        reconnectAttemptRef.current += 1;
        scheduleReconnect();
      }
    };

    connect();

    const staleDeviceSweep = window.setInterval(() => {
      pruneStaleDevices();
    }, DEVICE_ONLINE_SWEEP_MS);

    return () => {
      shouldReconnectRef.current = false;
      clearReconnectTimer();
      window.clearInterval(staleDeviceSweep);

      if (wsRef.current) {
        const socket = wsRef.current;
        wsRef.current = null;
        socket.close();
      }
    };
  }, [clientType, resolvedDeviceId]);

  const sendRFIDScan = useCallback((rfidUid: string, faceDescriptor?: number[]) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('[DeviceWS] WebSocket not connected');
      return;
    }

    wsRef.current.send(JSON.stringify({
      type: 'rfid_scan',
      rfidUid,
      faceDescriptor
    }));
  }, []);

  const sendRFIDDetected = useCallback((rfidUid: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('[DeviceWS] WebSocket not connected');
      return;
    }

    wsRef.current.send(JSON.stringify({
      type: 'rfid_detected',
      rfidUid,
    }));
  }, []);

  return {
    isConnected,
    deviceOnline,
    lastScanResult,
    sendRFIDScan,
    sendRFIDDetected,
    clearResult: () => setLastScanResult(null)
  };
}
