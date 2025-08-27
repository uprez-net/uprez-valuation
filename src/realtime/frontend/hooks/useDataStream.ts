/**
 * React Data Stream Hook
 * Handles real-time data subscriptions for valuation updates, market data, and calculations
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket, WebSocketMessage } from './useWebSocket';

export enum StreamType {
  VALUATION_UPDATE = 'valuation_update',
  MARKET_DATA = 'market_data',
  CALCULATION_PROGRESS = 'calculation_progress',
  FINANCIAL_METRICS = 'financial_metrics',
  CHART_DATA = 'chart_data',
  ERROR_NOTIFICATION = 'error_notification',
  STATUS_UPDATE = 'status_update'
}

export interface StreamData {
  stream_type: StreamType;
  data: any;
  timestamp: number;
  source: string;
  sequence_id: number;
  stream_id: string;
}

export interface ValuationUpdate {
  valuation_id: string;
  document_id: string;
  valuation: {
    enterprise_value: number;
    equity_value: number;
    price_per_share: number;
    market_cap: number;
    methodology: string;
    confidence_level: number;
  };
  update_type: 'calculation_complete' | 'parameter_change' | 'market_update';
  calculations: {
    dcf_value?: number;
    comparable_value?: number;
    asset_value?: number;
    weighted_average?: number;
  };
  timestamp: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap?: number;
  pe_ratio?: number;
  timestamp: number;
}

export interface CalculationProgress {
  calculation_id: string;
  calculation_type: string;
  progress: number; // 0-100
  status: string;
  intermediate_results?: any;
  elapsed_time: number;
  estimated_completion?: number;
}

export interface ChartData {
  chart_id: string;
  chart_type: 'line' | 'bar' | 'candlestick' | 'scatter';
  data_points: Array<{
    x: number | string;
    y: number;
    [key: string]: any;
  }>;
  metadata: {
    title: string;
    x_axis_label: string;
    y_axis_label: string;
    series_name: string;
  };
}

export interface ErrorNotification {
  error_id: string;
  error_type: 'validation' | 'calculation' | 'data' | 'system';
  message: string;
  details?: any;
  severity: 'low' | 'medium' | 'high' | 'critical';
  recoverable: boolean;
  suggested_actions?: string[];
}

export interface Subscription {
  subscription_id: string;
  stream_types: StreamType[];
  filters: Record<string, any>;
  created_at: number;
  active: boolean;
}

export interface UseDataStreamConfig {
  workspaceId: string;
  documentId?: string;
  autoSubscribe?: boolean;
  defaultStreamTypes?: StreamType[];
  bufferSize?: number;
  retryAttempts?: number;
}

export interface UseDataStreamReturn {
  // Subscriptions
  subscriptions: Subscription[];
  subscribe: (streamTypes: StreamType[], filters?: Record<string, any>) => Promise<string>;
  unsubscribe: (subscriptionId: string) => void;
  isSubscribed: (streamType: StreamType) => boolean;
  
  // Real-time data
  latestData: Record<StreamType, StreamData | null>;
  dataHistory: Record<StreamType, StreamData[]>;
  
  // Specific data types
  valuationUpdates: ValuationUpdate[];
  marketData: MarketData[];
  calculationProgress: CalculationProgress[];
  chartUpdates: ChartData[];
  errorNotifications: ErrorNotification[];
  
  // Stream management
  pauseStream: (streamType: StreamType) => void;
  resumeStream: (streamType: StreamType) => void;
  clearHistory: (streamType?: StreamType) => void;
  
  // State
  isLoading: boolean;
  error: string | null;
  connectionQuality: 'excellent' | 'good' | 'fair' | 'poor';
}

export const useDataStream = (
  webSocket: ReturnType<typeof useWebSocket>,
  config: UseDataStreamConfig
): UseDataStreamReturn => {
  const {
    workspaceId,
    documentId,
    autoSubscribe = true,
    defaultStreamTypes = [StreamType.VALUATION_UPDATE, StreamType.CALCULATION_PROGRESS],
    bufferSize = 100,
    retryAttempts = 3
  } = config;

  const { sendMessage, subscribe: wsSubscribe, isConnected } = webSocket;

  const [subscriptions, setSubscriptions] = useState<Subscription[]>([]);
  const [latestData, setLatestData] = useState<Record<StreamType, StreamData | null>>({
    [StreamType.VALUATION_UPDATE]: null,
    [StreamType.MARKET_DATA]: null,
    [StreamType.CALCULATION_PROGRESS]: null,
    [StreamType.FINANCIAL_METRICS]: null,
    [StreamType.CHART_DATA]: null,
    [StreamType.ERROR_NOTIFICATION]: null,
    [StreamType.STATUS_UPDATE]: null
  });
  const [dataHistory, setDataHistory] = useState<Record<StreamType, StreamData[]>>({
    [StreamType.VALUATION_UPDATE]: [],
    [StreamType.MARKET_DATA]: [],
    [StreamType.CALCULATION_PROGRESS]: [],
    [StreamType.FINANCIAL_METRICS]: [],
    [StreamType.CHART_DATA]: [],
    [StreamType.ERROR_NOTIFICATION]: [],
    [StreamType.STATUS_UPDATE]: []
  });

  const [valuationUpdates, setValuationUpdates] = useState<ValuationUpdate[]>([]);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [calculationProgress, setCalculationProgress] = useState<CalculationProgress[]>([]);
  const [chartUpdates, setChartUpdates] = useState<ChartData[]>([]);
  const [errorNotifications, setErrorNotifications] = useState<ErrorNotification[]>([]);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionQuality, setConnectionQuality] = useState<'excellent' | 'good' | 'fair' | 'poor'>('excellent');
  const [pausedStreams, setPausedStreams] = useState<Set<StreamType>>(new Set());

  const retryCountRef = useRef<Record<string, number>>({});
  const lastSequenceRef = useRef<Record<StreamType, number>>({});

  // Subscribe to stream types
  const subscribe = useCallback(async (streamTypes: StreamType[], filters: Record<string, any> = {}) => {
    if (!isConnected) {
      throw new Error('WebSocket not connected');
    }

    const subscriptionId = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    sendMessage('subscribe_stream', {
      subscription_id: subscriptionId,
      stream_types: streamTypes,
      filters: {
        ...filters,
        document_id: documentId,
        workspace_id: workspaceId
      }
    });

    const subscription: Subscription = {
      subscription_id: subscriptionId,
      stream_types: streamTypes,
      filters,
      created_at: Date.now(),
      active: true
    };

    setSubscriptions(prev => [...prev, subscription]);
    return subscriptionId;
  }, [isConnected, sendMessage, documentId, workspaceId]);

  // Unsubscribe from streams
  const unsubscribe = useCallback((subscriptionId: string) => {
    sendMessage('unsubscribe_stream', {
      subscription_id: subscriptionId
    });

    setSubscriptions(prev => prev.filter(sub => sub.subscription_id !== subscriptionId));
  }, [sendMessage]);

  // Check if subscribed to stream type
  const isSubscribed = useCallback((streamType: StreamType) => {
    return subscriptions.some(sub => 
      sub.active && sub.stream_types.includes(streamType)
    );
  }, [subscriptions]);

  // Pause/resume streams
  const pauseStream = useCallback((streamType: StreamType) => {
    setPausedStreams(prev => new Set([...prev, streamType]));
  }, []);

  const resumeStream = useCallback((streamType: StreamType) => {
    setPausedStreams(prev => {
      const newSet = new Set(prev);
      newSet.delete(streamType);
      return newSet;
    });
  }, []);

  // Clear history
  const clearHistory = useCallback((streamType?: StreamType) => {
    if (streamType) {
      setDataHistory(prev => ({
        ...prev,
        [streamType]: []
      }));
    } else {
      setDataHistory({
        [StreamType.VALUATION_UPDATE]: [],
        [StreamType.MARKET_DATA]: [],
        [StreamType.CALCULATION_PROGRESS]: [],
        [StreamType.FINANCIAL_METRICS]: [],
        [StreamType.CHART_DATA]: [],
        [StreamType.ERROR_NOTIFICATION]: [],
        [StreamType.STATUS_UPDATE]: []
      });
    }
  }, []);

  // Handle stream data messages
  useEffect(() => {
    const unsubscribe = wsSubscribe('stream_data', (message: WebSocketMessage) => {
      const streamData: StreamData = {
        stream_type: message.payload.stream_type as StreamType,
        data: message.payload.data,
        timestamp: message.payload.timestamp,
        source: message.payload.source,
        sequence_id: message.payload.sequence_id,
        stream_id: message.payload.stream_id || 'unknown'
      };

      // Check if stream is paused
      if (pausedStreams.has(streamData.stream_type)) {
        return;
      }

      // Check sequence to detect missed messages
      const lastSequence = lastSequenceRef.current[streamData.stream_type] || 0;
      if (streamData.sequence_id <= lastSequence) {
        // Duplicate or out-of-order message, skip
        return;
      }

      lastSequenceRef.current[streamData.stream_type] = streamData.sequence_id;

      // Update latest data
      setLatestData(prev => ({
        ...prev,
        [streamData.stream_type]: streamData
      }));

      // Update history with buffer limit
      setDataHistory(prev => {
        const newHistory = [...prev[streamData.stream_type], streamData];
        return {
          ...prev,
          [streamData.stream_type]: newHistory.slice(-bufferSize)
        };
      });

      // Update specific data arrays based on stream type
      switch (streamData.stream_type) {
        case StreamType.VALUATION_UPDATE:
          setValuationUpdates(prev => {
            const newUpdates = [...prev, streamData.data as ValuationUpdate];
            return newUpdates.slice(-50); // Keep last 50 valuation updates
          });
          break;

        case StreamType.MARKET_DATA:
          setMarketData(prev => {
            const newData = [...prev, streamData.data as MarketData];
            return newData.slice(-200); // Keep last 200 market data points
          });
          break;

        case StreamType.CALCULATION_PROGRESS:
          setCalculationProgress(prev => {
            const progress = streamData.data as CalculationProgress;
            const existingIndex = prev.findIndex(p => p.calculation_id === progress.calculation_id);
            
            if (existingIndex >= 0) {
              // Update existing calculation progress
              const updated = [...prev];
              updated[existingIndex] = progress;
              return updated;
            } else {
              // Add new calculation
              const newProgress = [...prev, progress];
              return newProgress.slice(-20); // Keep last 20 calculations
            }
          });
          break;

        case StreamType.CHART_DATA:
          setChartUpdates(prev => {
            const chart = streamData.data as ChartData;
            const existingIndex = prev.findIndex(c => c.chart_id === chart.chart_id);
            
            if (existingIndex >= 0) {
              // Update existing chart
              const updated = [...prev];
              updated[existingIndex] = chart;
              return updated;
            } else {
              // Add new chart
              const newCharts = [...prev, chart];
              return newCharts.slice(-10); // Keep last 10 charts
            }
          });
          break;

        case StreamType.ERROR_NOTIFICATION:
          setErrorNotifications(prev => {
            const error = streamData.data as ErrorNotification;
            const newErrors = [...prev, error];
            return newErrors.slice(-30); // Keep last 30 errors
          });
          break;
      }
    });

    return unsubscribe;
  }, [wsSubscribe, pausedStreams, bufferSize]);

  // Handle subscription confirmations
  useEffect(() => {
    const unsubscribe = wsSubscribe('subscription_confirmed', (message: WebSocketMessage) => {
      const { subscription_id } = message.payload;
      setIsLoading(false);
      setError(null);
      
      // Mark subscription as active
      setSubscriptions(prev => prev.map(sub => 
        sub.subscription_id === subscription_id 
          ? { ...sub, active: true }
          : sub
      ));
    });

    return unsubscribe;
  }, [wsSubscribe]);

  // Handle subscription errors
  useEffect(() => {
    const unsubscribe = wsSubscribe('subscription_error', (message: WebSocketMessage) => {
      const { subscription_id, error: errorMsg } = message.payload;
      setError(errorMsg);
      setIsLoading(false);
      
      // Mark subscription as inactive
      setSubscriptions(prev => prev.map(sub => 
        sub.subscription_id === subscription_id 
          ? { ...sub, active: false }
          : sub
      ));
    });

    return unsubscribe;
  }, [wsSubscribe]);

  // Monitor connection quality based on message timing
  useEffect(() => {
    let messageCount = 0;
    let lastMessageTime = Date.now();
    
    const checkConnectionQuality = () => {
      const now = Date.now();
      const timeSinceLastMessage = now - lastMessageTime;
      
      if (timeSinceLastMessage < 5000 && messageCount > 0) {
        setConnectionQuality('excellent');
      } else if (timeSinceLastMessage < 10000) {
        setConnectionQuality('good');
      } else if (timeSinceLastMessage < 30000) {
        setConnectionQuality('fair');
      } else {
        setConnectionQuality('poor');
      }
    };

    const unsubscribe = wsSubscribe('*', () => {
      messageCount++;
      lastMessageTime = Date.now();
    });

    const interval = setInterval(checkConnectionQuality, 5000);

    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [wsSubscribe]);

  // Auto-subscribe on connection
  useEffect(() => {
    if (isConnected && autoSubscribe && subscriptions.length === 0) {
      setIsLoading(true);
      subscribe(defaultStreamTypes).catch(err => {
        setError(err.message);
        setIsLoading(false);
      });
    }
  }, [isConnected, autoSubscribe, subscriptions.length, subscribe, defaultStreamTypes]);

  return {
    // Subscriptions
    subscriptions,
    subscribe,
    unsubscribe,
    isSubscribed,
    
    // Real-time data
    latestData,
    dataHistory,
    
    // Specific data types
    valuationUpdates,
    marketData,
    calculationProgress,
    chartUpdates,
    errorNotifications,
    
    // Stream management
    pauseStream,
    resumeStream,
    clearHistory,
    
    // State
    isLoading,
    error,
    connectionQuality
  };
};

export default useDataStream;