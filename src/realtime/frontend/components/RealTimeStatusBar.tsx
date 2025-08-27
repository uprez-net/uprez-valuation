/**
 * Real-Time Status Bar Component
 * Shows connection status, user presence indicators, and system health
 */

import React from 'react';
import { ConnectionStatus } from '../hooks/useWebSocket';
import { UserPresence } from '../hooks/useCollaboration';

interface RealTimeStatusBarProps {
  connectionStatus: ConnectionStatus;
  isConnected: boolean;
  presenceUsers: UserPresence[];
  currentUserId: string;
  connectionQuality: 'excellent' | 'good' | 'fair' | 'poor';
  messagesReceived: number;
  messagesSent: number;
  lastActivity?: number;
  onReconnect?: () => void;
  className?: string;
}

const RealTimeStatusBar: React.FC<RealTimeStatusBarProps> = ({
  connectionStatus,
  isConnected,
  presenceUsers,
  currentUserId,
  connectionQuality,
  messagesReceived,
  messagesSent,
  lastActivity,
  onReconnect,
  className = ''
}) => {
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case ConnectionStatus.CONNECTED:
        return 'text-green-600';
      case ConnectionStatus.CONNECTING:
      case ConnectionStatus.RECONNECTING:
        return 'text-yellow-600';
      case ConnectionStatus.DISCONNECTED:
      case ConnectionStatus.ERROR:
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getConnectionStatusIcon = () => {
    switch (connectionStatus) {
      case ConnectionStatus.CONNECTED:
        return '●';
      case ConnectionStatus.CONNECTING:
      case ConnectionStatus.RECONNECTING:
        return '◐';
      case ConnectionStatus.DISCONNECTED:
      case ConnectionStatus.ERROR:
        return '○';
      default:
        return '○';
    }
  };

  const getQualityColor = () => {
    switch (connectionQuality) {
      case 'excellent':
        return 'text-green-600';
      case 'good':
        return 'text-blue-600';
      case 'fair':
        return 'text-yellow-600';
      case 'poor':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getQualityBars = () => {
    const bars = [];
    const barCount = {
      excellent: 4,
      good: 3,
      fair: 2,
      poor: 1
    }[connectionQuality] || 0;

    for (let i = 0; i < 4; i++) {
      bars.push(
        <div
          key={i}
          className={`w-1 h-3 rounded-sm ${
            i < barCount ? getQualityColor().replace('text-', 'bg-') : 'bg-gray-300'
          }`}
        />
      );
    }
    return bars;
  };

  const formatLastActivity = () => {
    if (!lastActivity) return '';
    
    const diffMs = Date.now() - lastActivity;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffMs / 60000);

    if (diffSecs < 60) return `${diffSecs}s ago`;
    if (diffMins < 60) return `${diffMins}m ago`;
    return '1h+ ago';
  };

  const onlineUsers = presenceUsers.filter(u => u.status === 'online');
  const otherUsers = onlineUsers.filter(u => u.user_id !== currentUserId);

  return (
    <div className={`real-time-status-bar bg-gray-50 border-t border-gray-200 px-4 py-2 flex items-center justify-between text-sm ${className}`}>
      {/* Left Section: Connection Status */}
      <div className="flex items-center space-x-4">
        {/* Connection Indicator */}
        <div className="flex items-center space-x-2">
          <span className={`${getConnectionStatusColor()} font-medium`}>
            {getConnectionStatusIcon()}
          </span>
          <span className="text-gray-700 capitalize">
            {connectionStatus}
          </span>
          {!isConnected && onReconnect && (
            <button
              onClick={onReconnect}
              className="text-blue-600 hover:text-blue-800 underline ml-2"
            >
              Reconnect
            </button>
          )}
        </div>

        {/* Connection Quality */}
        {isConnected && (
          <div className="flex items-center space-x-2">
            <div className="flex space-x-px">
              {getQualityBars()}
            </div>
            <span className={`text-xs ${getQualityColor()}`}>
              {connectionQuality}
            </span>
          </div>
        )}

        {/* Message Stats */}
        {isConnected && (
          <div className="text-xs text-gray-500">
            ↓{messagesReceived} ↑{messagesSent}
          </div>
        )}
      </div>

      {/* Center Section: User Presence */}
      <div className="flex items-center space-x-3">
        {otherUsers.length > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-gray-600 text-xs">Active:</span>
            <div className="flex -space-x-1">
              {otherUsers.slice(0, 5).map((user) => (
                <div
                  key={user.user_id}
                  className="relative group"
                  title={`${user.display_name} - ${user.current_activity}`}
                >
                  <div
                    className="w-6 h-6 rounded-full border-2 border-white flex items-center justify-center text-xs font-medium text-white"
                    style={{ backgroundColor: user.color || '#6B7280' }}
                  >
                    {user.display_name.charAt(0).toUpperCase()}
                  </div>
                  <div className={`absolute -bottom-1 -right-1 w-2 h-2 rounded-full border border-white ${
                    user.status === 'online' ? 'bg-green-500' :
                    user.status === 'away' ? 'bg-yellow-500' :
                    user.status === 'busy' ? 'bg-red-500' :
                    'bg-gray-400'
                  }`} />
                  
                  {/* Tooltip */}
                  <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity z-50">
                    <div>{user.display_name}</div>
                    <div className="text-gray-300">{user.current_activity}</div>
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
                  </div>
                </div>
              ))}
              {otherUsers.length > 5 && (
                <div className="w-6 h-6 rounded-full bg-gray-400 border-2 border-white flex items-center justify-center text-xs font-medium text-white">
                  +{otherUsers.length - 5}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Current User Indicator */}
        <div className="flex items-center space-x-2">
          <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center text-xs font-medium text-white">
            {presenceUsers.find(u => u.user_id === currentUserId)?.display_name?.charAt(0).toUpperCase() || 'M'}
          </div>
          <span className="text-xs text-gray-600">You</span>
        </div>
      </div>

      {/* Right Section: System Info */}
      <div className="flex items-center space-x-4">
        {/* Last Activity */}
        {lastActivity && (
          <div className="text-xs text-gray-500">
            Last activity: {formatLastActivity()}
          </div>
        )}

        {/* Connection Time */}
        {isConnected && (
          <div className="text-xs text-gray-500">
            Connected
          </div>
        )}

        {/* System Status Indicator */}
        <div className="flex items-center space-x-1">
          <div className={`w-2 h-2 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`} />
          <span className="text-xs text-gray-600">
            {isConnected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default RealTimeStatusBar;