import { io, Socket } from 'socket.io-client';
import { WebSocketMessage } from '@/types';

class WebSocketClient {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(token: string): void {
    if (this.socket?.connected) {
      return;
    }

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
    
    this.socket = io(wsUrl, {
      auth: {
        token,
      },
      transports: ['websocket'],
      upgrade: true,
    });

    this.setupEventListeners();
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  private setupEventListeners(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('Connected to WebSocket server');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Disconnected from WebSocket server:', reason);
      
      // Auto-reconnect logic
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, don't reconnect
        return;
      }
      
      this.handleReconnection();
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.handleReconnection();
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    setTimeout(() => {
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      this.socket?.connect();
    }, delay);
  }

  // Join a project room for real-time collaboration
  joinProject(projectId: string): void {
    this.emit('join_project', { projectId });
  }

  leaveProject(projectId: string): void {
    this.emit('leave_project', { projectId });
  }

  // Send a message
  emit(event: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    }
  }

  // Subscribe to events
  on(event: string, callback: (data: any) => void): void {
    this.socket?.on(event, callback);
  }

  // Unsubscribe from events
  off(event: string, callback?: (data: any) => void): void {
    if (callback) {
      this.socket?.off(event, callback);
    } else {
      this.socket?.off(event);
    }
  }

  // Specific business event handlers
  onProjectUpdate(callback: (data: any) => void): void {
    this.on('project_update', callback);
  }

  onUserJoined(callback: (data: any) => void): void {
    this.on('user_joined', callback);
  }

  onUserLeft(callback: (data: any) => void): void {
    this.on('user_left', callback);
  }

  onDocumentProcessed(callback: (data: any) => void): void {
    this.on('document_processed', callback);
  }

  onCalculationComplete(callback: (data: any) => void): void {
    this.on('calculation_complete', callback);
  }

  onValuationUpdate(callback: (data: any) => void): void {
    this.on('valuation_update', callback);
  }

  // Send project updates
  sendProjectUpdate(projectId: string, update: any): void {
    this.emit('project_update', {
      projectId,
      update,
      timestamp: new Date().toISOString(),
    });
  }

  // Send cursor position for collaboration
  sendCursorPosition(projectId: string, position: { x: number; y: number; element?: string }): void {
    this.emit('cursor_position', {
      projectId,
      position,
      timestamp: new Date().toISOString(),
    });
  }

  // Get connection status
  get isConnected(): boolean {
    return this.socket?.connected || false;
  }

  get connectionId(): string | undefined {
    return this.socket?.id;
  }
}

export const wsClient = new WebSocketClient();