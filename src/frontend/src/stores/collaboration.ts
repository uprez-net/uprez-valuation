import { create } from 'zustand';
import { wsClient } from '@/lib/websocket';

interface CollaborationUser {
  id: string;
  name: string;
  avatar?: string;
  cursor?: {
    x: number;
    y: number;
    element?: string;
  };
  lastSeen: string;
}

interface CollaborationState {
  connectedUsers: CollaborationUser[];
  isConnected: boolean;
  currentProjectId: string | null;
  activities: Activity[];
  
  // Actions
  joinProject: (projectId: string) => void;
  leaveProject: () => void;
  updateCursor: (position: { x: number; y: number; element?: string }) => void;
  addActivity: (activity: Activity) => void;
  clearActivities: () => void;
}

interface Activity {
  id: string;
  userId: string;
  userName: string;
  action: string;
  details: string;
  timestamp: string;
}

export const useCollaborationStore = create<CollaborationState>((set, get) => ({
  connectedUsers: [],
  isConnected: false,
  currentProjectId: null,
  activities: [],

  joinProject: (projectId: string) => {
    const { currentProjectId } = get();
    
    // Leave current project if different
    if (currentProjectId && currentProjectId !== projectId) {
      wsClient.leaveProject(currentProjectId);
    }
    
    // Join new project
    wsClient.joinProject(projectId);
    set({ 
      currentProjectId: projectId,
      isConnected: wsClient.isConnected,
      connectedUsers: [],
      activities: [],
    });
    
    // Set up event listeners
    setupWebSocketListeners(set, get);
  },

  leaveProject: () => {
    const { currentProjectId } = get();
    
    if (currentProjectId) {
      wsClient.leaveProject(currentProjectId);
    }
    
    // Clean up event listeners
    wsClient.off('user_joined');
    wsClient.off('user_left');
    wsClient.off('cursor_position');
    wsClient.off('project_update');
    
    set({
      currentProjectId: null,
      connectedUsers: [],
      isConnected: false,
      activities: [],
    });
  },

  updateCursor: (position: { x: number; y: number; element?: string }) => {
    const { currentProjectId } = get();
    
    if (currentProjectId && wsClient.isConnected) {
      wsClient.sendCursorPosition(currentProjectId, position);
    }
  },

  addActivity: (activity: Activity) => {
    set((state) => ({
      activities: [activity, ...state.activities.slice(0, 49)], // Keep last 50 activities
    }));
  },

  clearActivities: () => {
    set({ activities: [] });
  },
}));

function setupWebSocketListeners(
  set: (partial: Partial<CollaborationState>) => void,
  get: () => CollaborationState
) {
  // User joined
  wsClient.on('user_joined', (data: { user: CollaborationUser }) => {
    set((state) => ({
      connectedUsers: [...state.connectedUsers.filter(u => u.id !== data.user.id), data.user],
    }));
    
    get().addActivity({
      id: `join_${data.user.id}_${Date.now()}`,
      userId: data.user.id,
      userName: data.user.name,
      action: 'joined',
      details: 'joined the project',
      timestamp: new Date().toISOString(),
    });
  });

  // User left
  wsClient.on('user_left', (data: { userId: string; userName: string }) => {
    set((state) => ({
      connectedUsers: state.connectedUsers.filter(u => u.id !== data.userId),
    }));
    
    get().addActivity({
      id: `leave_${data.userId}_${Date.now()}`,
      userId: data.userId,
      userName: data.userName,
      action: 'left',
      details: 'left the project',
      timestamp: new Date().toISOString(),
    });
  });

  // Cursor position update
  wsClient.on('cursor_position', (data: { 
    userId: string; 
    position: { x: number; y: number; element?: string }; 
  }) => {
    set((state) => ({
      connectedUsers: state.connectedUsers.map(user =>
        user.id === data.userId
          ? { ...user, cursor: data.position }
          : user
      ),
    }));
  });

  // Project updates
  wsClient.on('project_update', (data: { 
    userId: string; 
    userName: string; 
    update: any; 
  }) => {
    get().addActivity({
      id: `update_${data.userId}_${Date.now()}`,
      userId: data.userId,
      userName: data.userName,
      action: 'updated',
      details: getUpdateDescription(data.update),
      timestamp: new Date().toISOString(),
    });
  });

  // Document processed
  wsClient.on('document_processed', (data: { 
    documentName: string; 
    userId: string; 
    userName: string; 
  }) => {
    get().addActivity({
      id: `doc_${Date.now()}`,
      userId: data.userId,
      userName: data.userName,
      action: 'processed document',
      details: `processed "${data.documentName}"`,
      timestamp: new Date().toISOString(),
    });
  });

  // Calculation complete
  wsClient.on('calculation_complete', (data: { 
    scenarioName: string; 
    userId: string; 
    userName: string; 
  }) => {
    get().addActivity({
      id: `calc_${Date.now()}`,
      userId: data.userId,
      userName: data.userName,
      action: 'completed calculation',
      details: `calculated "${data.scenarioName}" scenario`,
      timestamp: new Date().toISOString(),
    });
  });
}

function getUpdateDescription(update: any): string {
  if (update.type === 'scenario') {
    return `modified ${update.scenarioName} scenario`;
  }
  if (update.type === 'metrics') {
    return 'updated financial metrics';
  }
  if (update.type === 'document') {
    return `uploaded ${update.documentName}`;
  }
  return 'made changes to the project';
}