/**
 * React WebSocket Hook for Real-time Collaboration
 * Provides WebSocket connection management, auto-reconnection, and message handling
 */

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';

export enum ConnectionStatus {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error'
}

export interface WebSocketMessage {
  type: string;
  payload: any;
  user_id: string;
  session_id: string;
  workspace_id: string;
  timestamp: number;
  message_id: string;
}

export interface WebSocketConfig {
  url: string;
  token: string;
  workspaceId: string;
  userId: string;
  sessionId: string;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
}

export interface UseWebSocketReturn {
  status: ConnectionStatus;
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  sendMessage: (type: string, payload: any) => void;
  subscribe: (messageType: string, handler: (message: WebSocketMessage) => void) => () => void;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  messageHistory: WebSocketMessage[];
  connectionStats: {
    connectedAt: number | null;
    reconnectCount: number;
    messagessent: number;
    messagesReceived: number;
  };
}

export const useWebSocket = (config: WebSocketConfig): UseWebSocketReturn => {
  const {
    url,
    token,
    workspaceId,
    userId,
    sessionId,
    autoReconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
    heartbeatInterval = 30000,
    messageQueueSize = 100
  } = config;

  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [messageHistory, setMessageHistory] = useState<WebSocketMessage[]>([]);
  const [connectionStats, setConnectionStats] = useState({
    connectedAt: null as number | null,
    reconnectCount: 0,
    messagesReceived: 0,
    messagesent: 0
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const messageHandlersRef = useRef<Map<string, Set<(message: WebSocketMessage) => void>>>(new Map());
  const messageQueueRef = useRef<Array<{ type: string; payload: any }>>([]);

  // Build WebSocket URL with query parameters
  const wsUrl = useMemo(() => {
    const baseUrl = url.replace('http', 'ws');
    const params = new URLSearchParams({
      token,
      session_id: sessionId,
      user_id: userId
    });
    return `${baseUrl}/ws/${workspaceId}?${params.toString()}`;
  }, [url, token, workspaceId, userId, sessionId]);

  // Send message with queuing support
  const sendMessage = useCallback((type: string, payload: any) => {
    const message = { type, payload };

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
        setConnectionStats(prev => ({
          ...prev,
          messagesent: prev.messagesent + 1
        }));
      } catch (error) {
        console.error('Failed to send message:', error);
        // Queue message for later sending
        messageQueueRef.current.push(message);
        if (messageQueueRef.current.length > messageQueueSize) {
          messageQueueRef.current.shift();
        }
      }
    } else {
      // Queue message if not connected
      messageQueueRef.current.push(message);
      if (messageQueueRef.current.length > messageQueueSize) {
        messageQueueRef.current.shift();
      }
    }
  }, [messageQueueSize]);

  // Process queued messages when connection is established
  const processQueuedMessages = useCallback(() => {
    while (messageQueueRef.current.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
      const message = messageQueueRef.current.shift()!;
      sendMessage(message.type, message.payload);
    }
  }, [sendMessage]);

  // Handle incoming messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      setLastMessage(message);
      setMessageHistory(prev => {
        const newHistory = [...prev, message];
        return newHistory.slice(-messageQueueSize); // Keep only recent messages
      });

      setConnectionStats(prev => ({
        ...prev,
        messagesReceived: prev.messagesReceived + 1
      }));

      // Call registered handlers
      const handlers = messageHandlersRef.current.get(message.type);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(message);
          } catch (error) {
            console.error(`Error in message handler for ${message.type}:`, error);
          }
        });
      }

      // Call wildcard handlers
      const wildcardHandlers = messageHandlersRef.current.get('*');
      if (wildcardHandlers) {
        wildcardHandlers.forEach(handler => {
          try {
            handler(message);
          } catch (error) {
            console.error('Error in wildcard message handler:', error);
          }
        });
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }, [messageQueueSize]);

  // Subscribe to message types
  const subscribe = useCallback((messageType: string, handler: (message: WebSocketMessage) => void) => {
    if (!messageHandlersRef.current.has(messageType)) {
      messageHandlersRef.current.set(messageType, new Set());
    }
    messageHandlersRef.current.get(messageType)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = messageHandlersRef.current.get(messageType);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          messageHandlersRef.current.delete(messageType);
        }
      }
    };
  }, []);

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
    }

    heartbeatTimeoutRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        sendMessage('heartbeat', { timestamp: Date.now() });
      }
    }, heartbeatInterval);
  }, [sendMessage, heartbeatInterval]);

  // Stop heartbeat
  const stopHeartbeat = useCallback(() => {
    if (heartbeatTimeoutRef.current) {
      clearInterval(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || 
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setStatus(ConnectionStatus.CONNECTING);

    try {
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setStatus(ConnectionStatus.CONNECTED);
        setConnectionStats(prev => ({
          ...prev,
          connectedAt: Date.now()
        }));
        
        reconnectAttemptsRef.current = 0;
        processQueuedMessages();
        startHeartbeat();

        // Send initial presence update
        sendMessage('presence_update', {
          status: 'online',
          activity: 'connected'
        });
      };

      wsRef.current.onmessage = handleMessage;

      wsRef.current.onclose = (event) => {
        stopHeartbeat();
        setStatus(ConnectionStatus.DISCONNECTED);
        setConnectionStats(prev => ({
          ...prev,
          connectedAt: null
        }));

        // Auto-reconnect if enabled and not a clean close
        if (autoReconnect && event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          setStatus(ConnectionStatus.RECONNECTING);
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++;
            setConnectionStats(prev => ({
              ...prev,
              reconnectCount: prev.reconnectCount + 1
            }));
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus(ConnectionStatus.ERROR);
        stopHeartbeat();
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setStatus(ConnectionStatus.ERROR);
    }
  }, [wsUrl, autoReconnect, maxReconnectAttempts, reconnectInterval, handleMessage, processQueuedMessages, startHeartbeat, stopHeartbeat, sendMessage]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    stopHeartbeat();

    if (wsRef.current) {
      // Send disconnect message
      if (wsRef.current.readyState === WebSocket.OPEN) {
        sendMessage('presence_update', {
          status: 'offline',
          activity: 'disconnected'
        });
      }
      
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }

    setStatus(ConnectionStatus.DISCONNECTED);
  }, [stopHeartbeat, sendMessage]);

  // Reconnect manually
  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttemptsRef.current = 0;
    setTimeout(connect, 100);
  }, [disconnect, connect]);

  // Initialize connection on mount
  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      stopHeartbeat();
    };
  }, [stopHeartbeat]);

  return {
    status,
    isConnected: status === ConnectionStatus.CONNECTED,
    lastMessage,
    sendMessage,
    subscribe,
    connect,
    disconnect,
    reconnect,
    messageHistory,
    connectionStats
  };
};

export default useWebSocket;