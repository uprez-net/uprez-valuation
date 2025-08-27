"""
Live Data Synchronization System
Handles real-time valuation updates, market data streaming, and calculation progress
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)

class StreamType(Enum):
    """Types of data streams"""
    VALUATION_UPDATE = "valuation_update"
    MARKET_DATA = "market_data"
    CALCULATION_PROGRESS = "calculation_progress"
    FINANCIAL_METRICS = "financial_metrics"
    CHART_DATA = "chart_data"
    ERROR_NOTIFICATION = "error_notification"
    STATUS_UPDATE = "status_update"

@dataclass
class StreamData:
    """Data container for streaming updates"""
    stream_type: StreamType
    data: Dict[str, Any]
    timestamp: float
    source: str
    workspace_id: str
    document_id: Optional[str] = None
    user_id: Optional[str] = None
    sequence_id: int = 0
    stream_id: str = ""
    
    def __post_init__(self):
        if not self.stream_id:
            self.stream_id = str(uuid4())

@dataclass
class Subscription:
    """User subscription to data streams"""
    user_id: str
    workspace_id: str
    stream_types: Set[StreamType]
    filters: Dict[str, Any]
    callback: Optional[Callable] = None
    created_at: float = 0
    last_update: float = 0
    active: bool = True
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        self.last_update = self.created_at

class DataStreamManager:
    """Manages real-time data streams and subscriptions"""
    
    def __init__(self):
        self.subscriptions: Dict[str, Subscription] = {}  # subscription_id -> Subscription
        self.user_subscriptions: Dict[str, Set[str]] = {}  # user_id -> set of subscription_ids
        self.workspace_subscriptions: Dict[str, Set[str]] = {}  # workspace_id -> set of subscription_ids
        self.stream_history: Dict[str, List[StreamData]] = {}  # stream_type -> recent data
        self.sequence_counters: Dict[str, int] = {}  # stream_type -> sequence counter
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
    
    def start(self):
        """Start the data stream manager"""
        self._running = True
        # Start background tasks
        task = asyncio.create_task(self._cleanup_old_data())
        self._tasks.add(task)
        logger.info("Data stream manager started")
    
    def stop(self):
        """Stop the data stream manager"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("Data stream manager stopped")
    
    def subscribe(self, user_id: str, workspace_id: str, stream_types: List[StreamType],
                 filters: Dict[str, Any] = None, callback: Callable = None) -> str:
        """Subscribe user to data streams"""
        subscription_id = str(uuid4())
        
        subscription = Subscription(
            user_id=user_id,
            workspace_id=workspace_id,
            stream_types=set(stream_types),
            filters=filters or {},
            callback=callback
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Track by user
        if user_id not in self.user_subscriptions:
            self.user_subscriptions[user_id] = set()
        self.user_subscriptions[user_id].add(subscription_id)
        
        # Track by workspace
        if workspace_id not in self.workspace_subscriptions:
            self.workspace_subscriptions[workspace_id] = set()
        self.workspace_subscriptions[workspace_id].add(subscription_id)
        
        logger.info(f"User {user_id} subscribed to {len(stream_types)} stream types")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription"""
        if subscription_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[subscription_id]
        
        # Remove from user tracking
        if subscription.user_id in self.user_subscriptions:
            self.user_subscriptions[subscription.user_id].discard(subscription_id)
            if not self.user_subscriptions[subscription.user_id]:
                del self.user_subscriptions[subscription.user_id]
        
        # Remove from workspace tracking
        if subscription.workspace_id in self.workspace_subscriptions:
            self.workspace_subscriptions[subscription.workspace_id].discard(subscription_id)
            if not self.workspace_subscriptions[subscription.workspace_id]:
                del self.workspace_subscriptions[subscription.workspace_id]
        
        del self.subscriptions[subscription_id]
        logger.info(f"Subscription {subscription_id} removed")
        return True
    
    def unsubscribe_user(self, user_id: str):
        """Remove all subscriptions for a user"""
        if user_id not in self.user_subscriptions:
            return
        
        subscription_ids = list(self.user_subscriptions[user_id])
        for subscription_id in subscription_ids:
            self.unsubscribe(subscription_id)
    
    async def publish(self, stream_data: StreamData) -> int:
        """Publish data to subscribers"""
        # Add sequence number
        stream_key = f"{stream_data.stream_type.value}:{stream_data.workspace_id}"
        if stream_key not in self.sequence_counters:
            self.sequence_counters[stream_key] = 0
        self.sequence_counters[stream_key] += 1
        stream_data.sequence_id = self.sequence_counters[stream_key]
        
        # Store in history
        history_key = stream_data.stream_type.value
        if history_key not in self.stream_history:
            self.stream_history[history_key] = []
        self.stream_history[history_key].append(stream_data)
        
        # Find matching subscriptions
        matching_subscriptions = self._find_matching_subscriptions(stream_data)
        
        # Send to subscribers
        sent_count = 0
        for subscription in matching_subscriptions:
            if subscription.callback:
                try:
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(stream_data)
                    else:
                        subscription.callback(stream_data)
                    subscription.last_update = time.time()
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Error calling subscription callback: {e}")
        
        logger.debug(f"Published {stream_data.stream_type.value} to {sent_count} subscribers")
        return sent_count
    
    def _find_matching_subscriptions(self, stream_data: StreamData) -> List[Subscription]:
        """Find subscriptions that match the stream data"""
        matching = []
        
        # Get workspace subscriptions
        workspace_subscription_ids = self.workspace_subscriptions.get(stream_data.workspace_id, set())
        
        for subscription_id in workspace_subscription_ids:
            subscription = self.subscriptions.get(subscription_id)
            if not subscription or not subscription.active:
                continue
            
            # Check stream type
            if stream_data.stream_type not in subscription.stream_types:
                continue
            
            # Check filters
            if self._matches_filters(stream_data, subscription.filters):
                matching.append(subscription)
        
        return matching
    
    def _matches_filters(self, stream_data: StreamData, filters: Dict[str, Any]) -> bool:
        """Check if stream data matches subscription filters"""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            if key == "document_id":
                if stream_data.document_id != expected_value:
                    return False
            elif key == "user_id":
                if stream_data.user_id != expected_value:
                    return False
            elif key == "source":
                if stream_data.source != expected_value:
                    return False
            elif key in stream_data.data:
                if stream_data.data[key] != expected_value:
                    return False
        
        return True
    
    def get_recent_data(self, stream_type: StreamType, limit: int = 10) -> List[StreamData]:
        """Get recent data for a stream type"""
        history_key = stream_type.value
        recent_data = self.stream_history.get(history_key, [])
        return recent_data[-limit:] if recent_data else []
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """Get subscription statistics"""
        active_subscriptions = [s for s in self.subscriptions.values() if s.active]
        
        stats = {
            "total_subscriptions": len(active_subscriptions),
            "subscriptions_by_workspace": {},
            "subscriptions_by_stream_type": {},
            "active_users": len(self.user_subscriptions),
            "active_workspaces": len(self.workspace_subscriptions)
        }
        
        for subscription in active_subscriptions:
            # By workspace
            workspace = subscription.workspace_id
            if workspace not in stats["subscriptions_by_workspace"]:
                stats["subscriptions_by_workspace"][workspace] = 0
            stats["subscriptions_by_workspace"][workspace] += 1
            
            # By stream type
            for stream_type in subscription.stream_types:
                stream_name = stream_type.value
                if stream_name not in stats["subscriptions_by_stream_type"]:
                    stats["subscriptions_by_stream_type"][stream_name] = 0
                stats["subscriptions_by_stream_type"][stream_name] += 1
        
        return stats
    
    async def _cleanup_old_data(self):
        """Background task to cleanup old stream data"""
        while self._running:
            try:
                current_time = time.time()
                cutoff_time = current_time - 3600  # Keep 1 hour of history
                
                for stream_type, data_list in self.stream_history.items():
                    # Remove old data
                    self.stream_history[stream_type] = [
                        data for data in data_list 
                        if data.timestamp > cutoff_time
                    ]
                
                # Remove inactive subscriptions
                inactive_subscriptions = [
                    sub_id for sub_id, sub in self.subscriptions.items()
                    if current_time - sub.last_update > 1800  # 30 minutes
                ]
                
                for sub_id in inactive_subscriptions:
                    self.unsubscribe(sub_id)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)

class ValuationStreamHandler:
    """Handles valuation-specific streaming data"""
    
    def __init__(self, data_stream_manager: DataStreamManager):
        self.stream_manager = data_stream_manager
    
    async def stream_valuation_update(self, workspace_id: str, document_id: str,
                                    valuation_data: Dict[str, Any], user_id: str = None):
        """Stream valuation calculation updates"""
        stream_data = StreamData(
            stream_type=StreamType.VALUATION_UPDATE,
            data={
                "valuation": valuation_data,
                "update_type": "calculation_complete"
            },
            timestamp=time.time(),
            source="valuation_engine",
            workspace_id=workspace_id,
            document_id=document_id,
            user_id=user_id
        )
        
        await self.stream_manager.publish(stream_data)
    
    async def stream_calculation_progress(self, workspace_id: str, document_id: str,
                                        progress_data: Dict[str, Any]):
        """Stream calculation progress updates"""
        stream_data = StreamData(
            stream_type=StreamType.CALCULATION_PROGRESS,
            data=progress_data,
            timestamp=time.time(),
            source="valuation_engine",
            workspace_id=workspace_id,
            document_id=document_id
        )
        
        await self.stream_manager.publish(stream_data)
    
    async def stream_market_data_update(self, workspace_id: str, market_data: Dict[str, Any]):
        """Stream real-time market data updates"""
        stream_data = StreamData(
            stream_type=StreamType.MARKET_DATA,
            data=market_data,
            timestamp=time.time(),
            source="market_data_provider",
            workspace_id=workspace_id
        )
        
        await self.stream_manager.publish(stream_data)
    
    async def stream_chart_update(self, workspace_id: str, document_id: str,
                                chart_data: Dict[str, Any]):
        """Stream financial chart updates"""
        stream_data = StreamData(
            stream_type=StreamType.CHART_DATA,
            data=chart_data,
            timestamp=time.time(),
            source="chart_generator",
            workspace_id=workspace_id,
            document_id=document_id
        )
        
        await self.stream_manager.publish(stream_data)
    
    async def stream_error_notification(self, workspace_id: str, error_data: Dict[str, Any],
                                      document_id: str = None, user_id: str = None):
        """Stream error notifications"""
        stream_data = StreamData(
            stream_type=StreamType.ERROR_NOTIFICATION,
            data=error_data,
            timestamp=time.time(),
            source="system",
            workspace_id=workspace_id,
            document_id=document_id,
            user_id=user_id
        )
        
        await self.stream_manager.publish(stream_data)

class FinancialMetricsStreamer:
    """Streams financial metrics and calculations"""
    
    def __init__(self, data_stream_manager: DataStreamManager):
        self.stream_manager = data_stream_manager
        self.active_calculations: Dict[str, Dict[str, Any]] = {}
    
    async def start_calculation_stream(self, workspace_id: str, document_id: str,
                                     calculation_type: str, params: Dict[str, Any]):
        """Start streaming calculation progress"""
        calculation_id = f"{document_id}_{calculation_type}_{int(time.time())}"
        
        self.active_calculations[calculation_id] = {
            "workspace_id": workspace_id,
            "document_id": document_id,
            "calculation_type": calculation_type,
            "params": params,
            "started_at": time.time(),
            "progress": 0,
            "status": "started"
        }
        
        # Initial progress update
        await self.update_calculation_progress(calculation_id, 0, "Initializing calculation...")
        
        return calculation_id
    
    async def update_calculation_progress(self, calculation_id: str, progress: float,
                                        status_message: str, intermediate_results: Dict[str, Any] = None):
        """Update calculation progress"""
        if calculation_id not in self.active_calculations:
            return
        
        calc_info = self.active_calculations[calculation_id]
        calc_info["progress"] = progress
        calc_info["status"] = status_message
        calc_info["last_update"] = time.time()
        
        progress_data = {
            "calculation_id": calculation_id,
            "calculation_type": calc_info["calculation_type"],
            "progress": progress,
            "status": status_message,
            "intermediate_results": intermediate_results or {},
            "elapsed_time": time.time() - calc_info["started_at"]
        }
        
        stream_data = StreamData(
            stream_type=StreamType.CALCULATION_PROGRESS,
            data=progress_data,
            timestamp=time.time(),
            source="calculation_engine",
            workspace_id=calc_info["workspace_id"],
            document_id=calc_info["document_id"]
        )
        
        await self.stream_manager.publish(stream_data)
    
    async def complete_calculation(self, calculation_id: str, final_results: Dict[str, Any]):
        """Complete calculation and stream final results"""
        if calculation_id not in self.active_calculations:
            return
        
        calc_info = self.active_calculations[calculation_id]
        
        # Final progress update
        await self.update_calculation_progress(calculation_id, 100, "Calculation completed", final_results)
        
        # Stream final results
        stream_data = StreamData(
            stream_type=StreamType.FINANCIAL_METRICS,
            data={
                "calculation_id": calculation_id,
                "calculation_type": calc_info["calculation_type"],
                "results": final_results,
                "completed_at": time.time(),
                "total_duration": time.time() - calc_info["started_at"]
            },
            timestamp=time.time(),
            source="calculation_engine",
            workspace_id=calc_info["workspace_id"],
            document_id=calc_info["document_id"]
        )
        
        await self.stream_manager.publish(stream_data)
        
        # Cleanup
        del self.active_calculations[calculation_id]
    
    async def stream_live_metrics(self, workspace_id: str, document_id: str,
                                metrics: Dict[str, Any]):
        """Stream live financial metrics"""
        stream_data = StreamData(
            stream_type=StreamType.FINANCIAL_METRICS,
            data={
                "metrics": metrics,
                "is_live": True
            },
            timestamp=time.time(),
            source="metrics_calculator",
            workspace_id=workspace_id,
            document_id=document_id
        )
        
        await self.stream_manager.publish(stream_data)

# Global instances
data_stream_manager = DataStreamManager()
valuation_stream_handler = ValuationStreamHandler(data_stream_manager)
financial_metrics_streamer = FinancialMetricsStreamer(data_stream_manager)