/**
 * React Collaboration Hook
 * Provides high-level collaboration features: comments, annotations, presence
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket, WebSocketMessage } from './useWebSocket';

export interface Comment {
  comment_id: string;
  document_id: string;
  user_id: string;
  content: string;
  position?: {
    line?: number;
    column?: number;
    start?: number;
    end?: number;
  };
  status: 'active' | 'resolved' | 'archived';
  created_at: number;
  updated_at: number;
  replies: CommentReply[];
  tags?: string[];
}

export interface CommentReply {
  reply_id: string;
  comment_id: string;
  user_id: string;
  content: string;
  created_at: number;
}

export interface Annotation {
  annotation_id: string;
  document_id: string;
  user_id: string;
  type: 'highlight' | 'note' | 'bookmark' | 'suggestion';
  content: string;
  position: {
    start: number;
    end: number;
    line?: number;
    column?: number;
  };
  style: {
    color?: string;
    backgroundColor?: string;
    fontWeight?: string;
  };
  created_at: number;
}

export interface UserPresence {
  user_id: string;
  username: string;
  display_name: string;
  status: 'online' | 'away' | 'busy' | 'offline';
  cursor_position?: {
    position: number;
    timestamp: number;
  };
  selection_range?: {
    start: number;
    end: number;
    timestamp: number;
  };
  current_activity: string;
  last_activity: number;
  avatar_url?: string;
  color?: string;
}

export interface Activity {
  activity_id: string;
  user_id: string;
  activity_type: string;
  description: string;
  target_id: string;
  created_at: number;
  data: any;
}

export interface UseCollaborationConfig {
  workspaceId: string;
  documentId?: string;
  userId: string;
  username: string;
  displayName: string;
}

export interface UseCollaborationReturn {
  // Comments
  comments: Comment[];
  addComment: (content: string, position?: any) => void;
  replyToComment: (commentId: string, content: string) => void;
  resolveComment: (commentId: string) => void;
  
  // Annotations
  annotations: Annotation[];
  addAnnotation: (type: string, content: string, position: any, style?: any) => void;
  updateAnnotation: (annotationId: string, updates: Partial<Annotation>) => void;
  deleteAnnotation: (annotationId: string) => void;
  
  // Presence
  presenceUsers: UserPresence[];
  updatePresence: (updates: Partial<UserPresence>) => void;
  updateCursor: (position: number) => void;
  updateSelection: (start: number, end: number) => void;
  
  // Activity
  activities: Activity[];
  
  // State
  isLoading: boolean;
  error: string | null;
}

export const useCollaboration = (
  webSocket: ReturnType<typeof useWebSocket>,
  config: UseCollaborationConfig
): UseCollaborationReturn => {
  const { workspaceId, documentId, userId, username, displayName } = config;
  const { sendMessage, subscribe, isConnected } = webSocket;

  const [comments, setComments] = useState<Comment[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [presenceUsers, setPresenceUsers] = useState<UserPresence[]>([]);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cursorTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const selectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Add comment
  const addComment = useCallback((content: string, position?: any) => {
    if (!documentId) return;
    
    sendMessage('comment_add', {
      document_id: documentId,
      content,
      position
    });
  }, [sendMessage, documentId]);

  // Reply to comment
  const replyToComment = useCallback((commentId: string, content: string) => {
    sendMessage('comment_reply', {
      comment_id: commentId,
      content
    });
  }, [sendMessage]);

  // Resolve comment
  const resolveComment = useCallback((commentId: string) => {
    sendMessage('comment_resolve', {
      comment_id: commentId
    });
  }, [sendMessage]);

  // Add annotation
  const addAnnotation = useCallback((type: string, content: string, position: any, style: any = {}) => {
    if (!documentId) return;
    
    sendMessage('annotation_create', {
      document_id: documentId,
      type,
      content,
      position,
      style
    });
  }, [sendMessage, documentId]);

  // Update annotation
  const updateAnnotation = useCallback((annotationId: string, updates: Partial<Annotation>) => {
    sendMessage('annotation_update', {
      annotation_id: annotationId,
      updates
    });
  }, [sendMessage]);

  // Delete annotation
  const deleteAnnotation = useCallback((annotationId: string) => {
    sendMessage('annotation_delete', {
      annotation_id: annotationId
    });
  }, [sendMessage]);

  // Update presence
  const updatePresence = useCallback((updates: Partial<UserPresence>) => {
    sendMessage('presence_update', {
      ...updates,
      document_id: documentId
    });
  }, [sendMessage, documentId]);

  // Update cursor with debouncing
  const updateCursor = useCallback((position: number) => {
    if (!documentId) return;

    if (cursorTimeoutRef.current) {
      clearTimeout(cursorTimeoutRef.current);
    }

    cursorTimeoutRef.current = setTimeout(() => {
      sendMessage('cursor_update', {
        document_id: documentId,
        position,
        timestamp: Date.now()
      });
    }, 100); // Debounce cursor updates
  }, [sendMessage, documentId]);

  // Update selection with debouncing
  const updateSelection = useCallback((start: number, end: number) => {
    if (!documentId) return;

    if (selectionTimeoutRef.current) {
      clearTimeout(selectionTimeoutRef.current);
    }

    selectionTimeoutRef.current = setTimeout(() => {
      sendMessage('selection_update', {
        document_id: documentId,
        start,
        end,
        timestamp: Date.now()
      });
    }, 150); // Debounce selection updates
  }, [sendMessage, documentId]);

  // Handle comment messages
  useEffect(() => {
    const unsubscribeCommentAdded = subscribe('comment_added', (message: WebSocketMessage) => {
      const { comment } = message.payload;
      setComments(prev => [...prev, comment]);
    });

    const unsubscribeCommentReply = subscribe('comment_reply_added', (message: WebSocketMessage) => {
      const { reply } = message.payload;
      setComments(prev => prev.map(comment => 
        comment.comment_id === reply.comment_id
          ? { ...comment, replies: [...comment.replies, reply], updated_at: Date.now() }
          : comment
      ));
    });

    const unsubscribeCommentResolved = subscribe('comment_resolved', (message: WebSocketMessage) => {
      const { comment_id } = message.payload;
      setComments(prev => prev.map(comment => 
        comment.comment_id === comment_id
          ? { ...comment, status: 'resolved' as const, updated_at: Date.now() }
          : comment
      ));
    });

    return () => {
      unsubscribeCommentAdded();
      unsubscribeCommentReply();
      unsubscribeCommentResolved();
    };
  }, [subscribe]);

  // Handle annotation messages
  useEffect(() => {
    const unsubscribeAnnotationCreated = subscribe('annotation_created', (message: WebSocketMessage) => {
      const { annotation } = message.payload;
      setAnnotations(prev => [...prev, annotation]);
    });

    const unsubscribeAnnotationUpdated = subscribe('annotation_updated', (message: WebSocketMessage) => {
      const { annotation_id, updates } = message.payload;
      setAnnotations(prev => prev.map(annotation => 
        annotation.annotation_id === annotation_id
          ? { ...annotation, ...updates }
          : annotation
      ));
    });

    const unsubscribeAnnotationDeleted = subscribe('annotation_deleted', (message: WebSocketMessage) => {
      const { annotation_id } = message.payload;
      setAnnotations(prev => prev.filter(annotation => annotation.annotation_id !== annotation_id));
    });

    return () => {
      unsubscribeAnnotationCreated();
      unsubscribeAnnotationUpdated();
      unsubscribeAnnotationDeleted();
    };
  }, [subscribe]);

  // Handle presence messages
  useEffect(() => {
    const unsubscribePresenceUpdate = subscribe('presence_updated', (message: WebSocketMessage) => {
      const { user_id, status, document_id: msgDocId, activity, timestamp } = message.payload;
      
      setPresenceUsers(prev => {
        const existingIndex = prev.findIndex(user => user.user_id === user_id);
        const updates = {
          status,
          current_activity: activity,
          last_activity: timestamp
        };

        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = { ...updated[existingIndex], ...updates };
          return updated;
        } else {
          // New user - this would typically come from initial presence data
          return prev;
        }
      });
    });

    const unsubscribeCursorUpdate = subscribe('cursor_updated', (message: WebSocketMessage) => {
      const { user_id, position, timestamp } = message.payload;
      
      setPresenceUsers(prev => prev.map(user => 
        user.user_id === user_id
          ? { ...user, cursor_position: { position, timestamp } }
          : user
      ));
    });

    const unsubscribeSelectionUpdate = subscribe('selection_updated', (message: WebSocketMessage) => {
      const { user_id, start, end, timestamp } = message.payload;
      
      setPresenceUsers(prev => prev.map(user => 
        user.user_id === user_id
          ? { ...user, selection_range: { start, end, timestamp } }
          : user
      ));
    });

    const unsubscribePresenceJoined = subscribe('presence_update', (message: WebSocketMessage) => {
      const { action, user_id: msgUserId, metadata } = message.payload;
      
      if (action === 'joined' && msgUserId !== userId) {
        const newUser: UserPresence = {
          user_id: msgUserId,
          username: metadata?.username || msgUserId,
          display_name: metadata?.display_name || metadata?.username || msgUserId,
          status: 'online',
          current_activity: 'joined',
          last_activity: Date.now(),
          color: generateUserColor(msgUserId)
        };
        
        setPresenceUsers(prev => {
          const existing = prev.find(u => u.user_id === msgUserId);
          return existing ? prev : [...prev, newUser];
        });
      } else if (action === 'left') {
        setPresenceUsers(prev => prev.filter(user => user.user_id !== msgUserId));
      }
    });

    return () => {
      unsubscribePresenceUpdate();
      unsubscribeCursorUpdate();
      unsubscribeSelectionUpdate();
      unsubscribePresenceJoined();
    };
  }, [subscribe, userId]);

  // Load initial data when connected
  useEffect(() => {
    if (isConnected && documentId) {
      setIsLoading(true);
      
      // Request initial data
      sendMessage('get_initial_data', {
        document_id: documentId
      });

      // Set loading timeout
      const timeout = setTimeout(() => {
        setIsLoading(false);
      }, 5000);

      const unsubscribe = subscribe('initial_data', (message: WebSocketMessage) => {
        const { comments: initialComments, annotations: initialAnnotations, presence } = message.payload;
        
        setComments(initialComments || []);
        setAnnotations(initialAnnotations || []);
        setPresenceUsers(presence || []);
        setIsLoading(false);
        clearTimeout(timeout);
      });

      return () => {
        clearTimeout(timeout);
        unsubscribe();
      };
    }
  }, [isConnected, documentId, sendMessage, subscribe]);

  // Send initial presence when connected
  useEffect(() => {
    if (isConnected) {
      updatePresence({
        username,
        display_name: displayName,
        status: 'online',
        current_activity: 'viewing'
      });
    }
  }, [isConnected, updatePresence, username, displayName]);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (cursorTimeoutRef.current) {
        clearTimeout(cursorTimeoutRef.current);
      }
      if (selectionTimeoutRef.current) {
        clearTimeout(selectionTimeoutRef.current);
      }
    };
  }, []);

  return {
    // Comments
    comments,
    addComment,
    replyToComment,
    resolveComment,
    
    // Annotations
    annotations,
    addAnnotation,
    updateAnnotation,
    deleteAnnotation,
    
    // Presence
    presenceUsers,
    updatePresence,
    updateCursor,
    updateSelection,
    
    // Activity
    activities,
    
    // State
    isLoading,
    error
  };
};

// Helper function to generate consistent colors for users
const generateUserColor = (userId: string): string => {
  const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8E8', '#F7DC6F', '#BB8FCE', '#85C1E9'
  ];
  
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    hash = userId.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  return colors[Math.abs(hash) % colors.length];
};

export default useCollaboration;