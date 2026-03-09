import { useEffect, useRef, useState, useCallback } from 'react';

export interface DeviceScanResult {
  type: 'scan_result' | 'connected' | 'error';
  success?: boolean;
  message: string;
  employee?: { id: number; name: string };
  action?: 'ENTRY' | 'EXIT';
  matchConfidence?: number;
  rfidUid?: string;
}

export function useDeviceWS(deviceId: string = `web-${Date.now()}`) {
  const wsRef = useRef<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastScanResult, setLastScanResult] = useState<DeviceScanResult | null>(null);

  useEffect(() => {
    // Get the WebSocket protocol
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/device?deviceId=${encodeURIComponent(deviceId)}`;

    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('[DeviceWS] Connected to device server');
        setIsConnected(true);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as DeviceScanResult;
          setLastScanResult(message);
          console.log('[DeviceWS] Received:', message);
        } catch (error) {
          console.error('[DeviceWS] Error parsing message:', error);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('[DeviceWS] Error:', error);
        setIsConnected(false);
      };

      wsRef.current.onclose = () => {
        console.log('[DeviceWS] Disconnected');
        setIsConnected(false);
      };
    } catch (error) {
      console.error('[DeviceWS] Failed to create WebSocket:', error);
      setIsConnected(false);
    }

    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [deviceId]);

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

  return {
    isConnected,
    lastScanResult,
    sendRFIDScan,
    clearResult: () => setLastScanResult(null)
  };
}
