/**
 * Collaboration Panel Component
 * Displays comments, annotations, user presence, and activity feed
 */

import React, { useState, useRef, useEffect } from 'react';
import { Comment, Annotation, UserPresence, Activity } from '../hooks/useCollaboration';

interface CollaborationPanelProps {
  comments: Comment[];
  annotations: Annotation[];
  presenceUsers: UserPresence[];
  activities: Activity[];
  currentUserId: string;
  onAddComment: (content: string, position?: any) => void;
  onReplyToComment: (commentId: string, content: string) => void;
  onResolveComment: (commentId: string) => void;
  onAddAnnotation: (type: string, content: string, position: any, style?: any) => void;
  onDeleteAnnotation: (annotationId: string) => void;
  className?: string;
}

type TabType = 'comments' | 'annotations' | 'presence' | 'activity';

const CollaborationPanel: React.FC<CollaborationPanelProps> = ({
  comments,
  annotations,
  presenceUsers,
  activities,
  currentUserId,
  onAddComment,
  onReplyToComment,
  onResolveComment,
  onAddAnnotation,
  onDeleteAnnotation,
  className = ''
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('comments');
  const [newCommentContent, setNewCommentContent] = useState('');
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [replyContent, setReplyContent] = useState('');
  const [selectedPosition, setSelectedPosition] = useState<any>(null);
  
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const replyTextAreaRef = useRef<HTMLTextAreaElement>(null);

  // Focus textarea when replying
  useEffect(() => {
    if (replyingTo && replyTextAreaRef.current) {
      replyTextAreaRef.current.focus();
    }
  }, [replyingTo]);

  const handleAddComment = () => {
    if (newCommentContent.trim()) {
      onAddComment(newCommentContent, selectedPosition);
      setNewCommentContent('');
      setSelectedPosition(null);
    }
  };

  const handleReply = () => {
    if (replyContent.trim() && replyingTo) {
      onReplyToComment(replyingTo, replyContent);
      setReplyContent('');
      setReplyingTo(null);
    }
  };

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const getActiveComments = () => comments.filter(c => c.status === 'active');
  const getResolvedComments = () => comments.filter(c => c.status === 'resolved');
  const getOnlineUsers = () => presenceUsers.filter(u => u.status === 'online');

  const tabs = [
    { id: 'comments' as TabType, label: 'Comments', count: getActiveComments().length },
    { id: 'annotations' as TabType, label: 'Annotations', count: annotations.length },
    { id: 'presence' as TabType, label: 'Users', count: getOnlineUsers().length },
    { id: 'activity' as TabType, label: 'Activity', count: activities.length }
  ];

  return (
    <div className={`collaboration-panel bg-white border-l border-gray-200 w-80 flex flex-col h-full ${className}`}>
      {/* Tab Header */}
      <div className="flex border-b border-gray-200">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-3 py-2 text-sm font-medium border-b-2 ${
              activeTab === tab.id
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            {tab.label}
            {tab.count > 0 && (
              <span className="ml-1 px-1.5 py-0.5 text-xs bg-gray-100 text-gray-600 rounded-full">
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'comments' && (
          <div className="h-full flex flex-col">
            {/* Add Comment */}
            <div className="p-4 border-b border-gray-100">
              <div className="space-y-3">
                <textarea
                  ref={textAreaRef}
                  value={newCommentContent}
                  onChange={(e) => setNewCommentContent(e.target.value)}
                  placeholder="Add a comment..."
                  className="w-full p-2 border border-gray-300 rounded-md resize-none text-sm"
                  rows={3}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                      e.preventDefault();
                      handleAddComment();
                    }
                  }}
                />
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">
                    {selectedPosition ? 'Position selected' : 'No position selected'}
                  </span>
                  <button
                    onClick={handleAddComment}
                    disabled={!newCommentContent.trim()}
                    className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Add Comment
                  </button>
                </div>
              </div>
            </div>

            {/* Comments List */}
            <div className="flex-1 overflow-y-auto">
              {/* Active Comments */}
              <div className="p-4 space-y-4">
                {getActiveComments().map((comment) => (
                  <div key={comment.comment_id} className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-xs font-medium">
                          {comment.user_id.charAt(0).toUpperCase()}
                        </div>
                        <span className="text-sm font-medium text-gray-900">{comment.user_id}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <span className="text-xs text-gray-500">{formatTimestamp(comment.created_at)}</span>
                        <button
                          onClick={() => onResolveComment(comment.comment_id)}
                          className="text-xs text-green-600 hover:text-green-800"
                        >
                          Resolve
                        </button>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-700 mb-2">{comment.content}</p>
                    
                    {comment.position && (
                      <div className="text-xs text-gray-500 mb-2">
                        Position: {JSON.stringify(comment.position)}
                      </div>
                    )}

                    {/* Replies */}
                    {comment.replies.length > 0 && (
                      <div className="mt-3 space-y-2 pl-4 border-l-2 border-gray-200">
                        {comment.replies.map((reply) => (
                          <div key={reply.reply_id} className="bg-white rounded p-2">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium text-gray-800">{reply.user_id}</span>
                              <span className="text-xs text-gray-500">{formatTimestamp(reply.created_at)}</span>
                            </div>
                            <p className="text-sm text-gray-700">{reply.content}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Reply Box */}
                    {replyingTo === comment.comment_id ? (
                      <div className="mt-3 space-y-2">
                        <textarea
                          ref={replyTextAreaRef}
                          value={replyContent}
                          onChange={(e) => setReplyContent(e.target.value)}
                          placeholder="Write a reply..."
                          className="w-full p-2 border border-gray-300 rounded text-sm resize-none"
                          rows={2}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                              e.preventDefault();
                              handleReply();
                            }
                            if (e.key === 'Escape') {
                              setReplyingTo(null);
                              setReplyContent('');
                            }
                          }}
                        />
                        <div className="flex justify-end space-x-2">
                          <button
                            onClick={() => {
                              setReplyingTo(null);
                              setReplyContent('');
                            }}
                            className="px-2 py-1 text-xs text-gray-600 hover:text-gray-800"
                          >
                            Cancel
                          </button>
                          <button
                            onClick={handleReply}
                            disabled={!replyContent.trim()}
                            className="px-3 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600 disabled:opacity-50"
                          >
                            Reply
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={() => setReplyingTo(comment.comment_id)}
                        className="mt-2 text-xs text-blue-600 hover:text-blue-800"
                      >
                        Reply
                      </button>
                    )}
                  </div>
                ))}

                {getActiveComments().length === 0 && (
                  <div className="text-center text-gray-500 text-sm py-8">
                    No active comments yet. Add one to start collaborating!
                  </div>
                )}
              </div>

              {/* Resolved Comments */}
              {getResolvedComments().length > 0 && (
                <div className="border-t border-gray-200 p-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-3">Resolved Comments</h4>
                  <div className="space-y-2">
                    {getResolvedComments().map((comment) => (
                      <div key={comment.comment_id} className="bg-green-50 rounded p-2 opacity-75">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs font-medium text-gray-700">{comment.user_id}</span>
                          <span className="text-xs text-green-600">✓ Resolved</span>
                        </div>
                        <p className="text-sm text-gray-600">{comment.content}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'annotations' && (
          <div className="h-full overflow-y-auto p-4">
            <div className="space-y-3">
              {annotations.map((annotation) => (
                <div key={annotation.annotation_id} className="border border-gray-200 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        annotation.type === 'highlight' ? 'bg-yellow-100 text-yellow-800' :
                        annotation.type === 'note' ? 'bg-blue-100 text-blue-800' :
                        annotation.type === 'bookmark' ? 'bg-green-100 text-green-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {annotation.type}
                      </span>
                      <span className="text-sm text-gray-600">{annotation.user_id}</span>
                    </div>
                    <button
                      onClick={() => onDeleteAnnotation(annotation.annotation_id)}
                      className="text-xs text-red-600 hover:text-red-800"
                    >
                      Delete
                    </button>
                  </div>
                  
                  <p className="text-sm text-gray-700 mb-2">{annotation.content}</p>
                  
                  <div className="text-xs text-gray-500">
                    Position: {annotation.position.start} - {annotation.position.end}
                  </div>
                  
                  <div className="text-xs text-gray-400 mt-1">
                    {formatTimestamp(annotation.created_at)}
                  </div>
                </div>
              ))}

              {annotations.length === 0 && (
                <div className="text-center text-gray-500 text-sm py-8">
                  No annotations yet. Select text and add annotations to highlight important sections.
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'presence' && (
          <div className="h-full overflow-y-auto p-4">
            <div className="space-y-3">
              {presenceUsers.map((user) => (
                <div key={user.user_id} className="flex items-center space-x-3">
                  <div className="relative">
                    <div 
                      className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium"
                      style={{ backgroundColor: user.color || '#6B7280' }}
                    >
                      {user.display_name.charAt(0).toUpperCase()}
                    </div>
                    <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${
                      user.status === 'online' ? 'bg-green-500' :
                      user.status === 'away' ? 'bg-yellow-500' :
                      user.status === 'busy' ? 'bg-red-500' :
                      'bg-gray-400'
                    }`} />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {user.display_name}
                        {user.user_id === currentUserId && ' (You)'}
                      </p>
                      <span className="text-xs text-gray-500">{user.status}</span>
                    </div>
                    <p className="text-xs text-gray-500 truncate">{user.current_activity}</p>
                    {user.cursor_position && (
                      <p className="text-xs text-blue-600">Cursor at position {user.cursor_position.position}</p>
                    )}
                  </div>
                </div>
              ))}

              {presenceUsers.length === 0 && (
                <div className="text-center text-gray-500 text-sm py-8">
                  No other users currently online.
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'activity' && (
          <div className="h-full overflow-y-auto p-4">
            <div className="space-y-3">
              {activities.map((activity) => (
                <div key={activity.activity_id} className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-gray-300 rounded-full flex items-center justify-center text-white text-xs font-medium">
                    {activity.user_id.charAt(0).toUpperCase()}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900">{activity.description}</p>
                    <p className="text-xs text-gray-500">{formatTimestamp(activity.created_at)}</p>
                  </div>
                </div>
              ))}

              {activities.length === 0 && (
                <div className="text-center text-gray-500 text-sm py-8">
                  No recent activity.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CollaborationPanel;