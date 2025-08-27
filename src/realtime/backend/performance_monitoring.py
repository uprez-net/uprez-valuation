"""
Performance Monitoring and Optimization Module
Tracks real-time system performance, identifies bottlenecks, and provides scaling insights
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from uuid import uuid4
import json
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_name: str
    value: float
    timestamp: float
    component: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SystemHealth:
    """System health snapshot"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    active_connections: int
    message_rate: float
    error_rate: float
    response_time: float
    timestamp: float

@dataclass
class Bottleneck:
    """Performance bottleneck identification"""
    bottleneck_id: str
    component: str
    severity: str  # low, medium, high, critical
    description: str
    metrics: Dict[str, float]
    suggested_actions: List[str]
    detected_at: float
    resolved_at: Optional[float] = None

class PerformanceTracker:
    """Tracks performance metrics and identifies issues"""
    
    def __init__(self, metric_retention_seconds: int = 3600):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_retention_seconds = metric_retention_seconds
        self.bottlenecks: Dict[str, Bottleneck] = {}
        self._running = False
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage': 90.0,
            'response_time': 1000.0,  # milliseconds
            'error_rate': 5.0,  # percent
            'message_queue_size': 1000,
            'connection_count': 10000
        }
    
    def record_metric(self, metric_name: str, value: float, component: str = "system", 
                     metadata: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            component=component,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics[metric_name].append(metric)
            
        # Check for bottlenecks
        self._check_bottleneck(metric)
    
    def get_metrics(self, metric_name: str, time_range_seconds: int = 300) -> List[PerformanceMetric]:
        """Get metrics within time range"""
        cutoff_time = time.time() - time_range_seconds
        
        with self._lock:
            metrics = list(self.metrics[metric_name])
        
        return [m for m in metrics if m.timestamp >= cutoff_time]
    
    def get_average(self, metric_name: str, time_range_seconds: int = 300) -> Optional[float]:
        """Get average value for a metric"""
        metrics = self.get_metrics(metric_name, time_range_seconds)
        if not metrics:
            return None
        
        return sum(m.value for m in metrics) / len(metrics)
    
    def get_percentile(self, metric_name: str, percentile: float, 
                      time_range_seconds: int = 300) -> Optional[float]:
        """Get percentile value for a metric"""
        metrics = self.get_metrics(metric_name, time_range_seconds)
        if not metrics:
            return None
        
        values = sorted(m.value for m in metrics)
        index = int(percentile / 100.0 * len(values))
        return values[min(index, len(values) - 1)]
    
    def _check_bottleneck(self, metric: PerformanceMetric):
        """Check if metric indicates a bottleneck"""
        threshold = self.thresholds.get(metric.metric_name)
        if not threshold:
            return
        
        if metric.value > threshold:
            bottleneck_id = f"{metric.component}_{metric.metric_name}_{int(time.time())}"
            
            # Check if similar bottleneck already exists
            existing = None
            for bid, bottleneck in self.bottlenecks.items():
                if (bottleneck.component == metric.component and 
                    bottleneck.resolved_at is None and
                    metric.metric_name in str(bottleneck.description)):
                    existing = bottleneck
                    break
            
            if not existing:
                severity = self._determine_severity(metric.metric_name, metric.value, threshold)
                actions = self._get_suggested_actions(metric.metric_name, metric.component)
                
                bottleneck = Bottleneck(
                    bottleneck_id=bottleneck_id,
                    component=metric.component,
                    severity=severity,
                    description=f"High {metric.metric_name}: {metric.value:.2f} (threshold: {threshold})",
                    metrics={metric.metric_name: metric.value},
                    suggested_actions=actions,
                    detected_at=time.time()
                )
                
                self.bottlenecks[bottleneck_id] = bottleneck
                logger.warning(f"Bottleneck detected: {bottleneck.description}")
    
    def _determine_severity(self, metric_name: str, value: float, threshold: float) -> str:
        """Determine bottleneck severity"""
        ratio = value / threshold
        
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.2:
            return "medium"
        else:
            return "low"
    
    def _get_suggested_actions(self, metric_name: str, component: str) -> List[str]:
        """Get suggested actions for bottleneck"""
        actions = []
        
        if metric_name == "cpu_percent":
            actions = [
                "Scale horizontally by adding more server instances",
                "Optimize CPU-intensive operations",
                "Consider using async processing for heavy tasks",
                "Review and optimize database queries"
            ]
        elif metric_name == "memory_percent":
            actions = [
                "Increase server memory capacity",
                "Optimize memory usage in applications",
                "Implement memory caching strategies",
                "Review object lifecycle and garbage collection"
            ]
        elif metric_name == "response_time":
            actions = [
                "Optimize database queries",
                "Implement response caching",
                "Add CDN for static assets",
                "Review and optimize critical code paths"
            ]
        elif metric_name == "error_rate":
            actions = [
                "Review error logs for common issues",
                "Implement better error handling",
                "Add input validation and sanitization",
                "Monitor third-party service dependencies"
            ]
        
        return actions
    
    def resolve_bottleneck(self, bottleneck_id: str):
        """Mark bottleneck as resolved"""
        if bottleneck_id in self.bottlenecks:
            self.bottlenecks[bottleneck_id].resolved_at = time.time()
            logger.info(f"Bottleneck resolved: {bottleneck_id}")
    
    def get_active_bottlenecks(self) -> List[Bottleneck]:
        """Get all unresolved bottlenecks"""
        return [b for b in self.bottlenecks.values() if b.resolved_at is None]
    
    def cleanup_old_data(self):
        """Remove old metrics and resolved bottlenecks"""
        cutoff_time = time.time() - self.metric_retention_seconds
        
        with self._lock:
            for metric_name, metric_queue in self.metrics.items():
                # Remove old metrics
                while metric_queue and metric_queue[0].timestamp < cutoff_time:
                    metric_queue.popleft()
        
        # Remove old resolved bottlenecks
        old_bottlenecks = [
            bid for bid, bottleneck in self.bottlenecks.items()
            if (bottleneck.resolved_at and 
                bottleneck.resolved_at < cutoff_time)
        ]
        
        for bid in old_bottlenecks:
            del self.bottlenecks[bid]

class SystemMonitor:
    """System-wide performance monitoring"""
    
    def __init__(self, performance_tracker: PerformanceTracker):
        self.tracker = performance_tracker
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.last_network_io: Optional[Dict[str, int]] = None
    
    async def start(self, interval_seconds: float = 5.0):
        """Start system monitoring"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))
        logger.info("System monitor started")
    
    async def stop(self):
        """Stop system monitoring"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitor stopped")
    
    async def _monitor_loop(self, interval_seconds: float):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.tracker.record_metric("cpu_percent", cpu_percent, "system")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.tracker.record_metric("memory_percent", memory.percent, "system")
            self.tracker.record_metric("memory_available", memory.available, "system")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.tracker.record_metric("disk_usage", disk_percent, "system")
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if self.last_network_io:
                bytes_sent_rate = network_io.bytes_sent - self.last_network_io.get('bytes_sent', 0)
                bytes_recv_rate = network_io.bytes_recv - self.last_network_io.get('bytes_recv', 0)
                
                self.tracker.record_metric("network_bytes_sent_rate", bytes_sent_rate, "system")
                self.tracker.record_metric("network_bytes_recv_rate", bytes_recv_rate, "system")
            
            self.last_network_io = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv
            }
            
            # Process-specific metrics
            process = psutil.Process()
            self.tracker.record_metric("process_memory_rss", process.memory_info().rss, "process")
            self.tracker.record_metric("process_cpu_percent", process.cpu_percent(), "process")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

class WebSocketPerformanceMonitor:
    """WebSocket-specific performance monitoring"""
    
    def __init__(self, performance_tracker: PerformanceTracker):
        self.tracker = performance_tracker
        self.message_timestamps: deque = deque(maxlen=1000)
        self.error_count = 0
        self.total_messages = 0
        self.response_times: deque = deque(maxlen=100)
    
    def record_message_sent(self):
        """Record WebSocket message sent"""
        self.message_timestamps.append(time.time())
        self.total_messages += 1
        
        # Calculate message rate (messages per second)
        current_time = time.time()
        recent_messages = [t for t in self.message_timestamps if current_time - t <= 60]
        message_rate = len(recent_messages) / 60.0
        
        self.tracker.record_metric("websocket_message_rate", message_rate, "websocket")
        self.tracker.record_metric("websocket_total_messages", self.total_messages, "websocket")
    
    def record_message_error(self):
        """Record WebSocket message error"""
        self.error_count += 1
        error_rate = (self.error_count / max(self.total_messages, 1)) * 100
        
        self.tracker.record_metric("websocket_error_rate", error_rate, "websocket")
        self.tracker.record_metric("websocket_error_count", self.error_count, "websocket")
    
    def record_response_time(self, response_time_ms: float):
        """Record WebSocket response time"""
        self.response_times.append(response_time_ms)
        
        if len(self.response_times) >= 10:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            self.tracker.record_metric("websocket_avg_response_time", avg_response_time, "websocket")
    
    def record_connection_count(self, count: int):
        """Record active WebSocket connection count"""
        self.tracker.record_metric("websocket_connection_count", count, "websocket")

class PerformanceAnalyzer:
    """Analyzes performance trends and provides insights"""
    
    def __init__(self, performance_tracker: PerformanceTracker):
        self.tracker = performance_tracker
    
    def analyze_trends(self, metric_name: str, time_range_seconds: int = 3600) -> Dict[str, Any]:
        """Analyze performance trends for a metric"""
        metrics = self.tracker.get_metrics(metric_name, time_range_seconds)
        if len(metrics) < 2:
            return {"status": "insufficient_data"}
        
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        # Calculate trend
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "status": "analyzed",
            "trend": trend,
            "slope": slope,
            "current_value": values[-1],
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "data_points": len(values),
            "time_range": time_range_seconds
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        summary = {
            "timestamp": time.time(),
            "metrics": {},
            "bottlenecks": [],
            "health_score": 100,
            "recommendations": []
        }
        
        # Key metrics summary
        key_metrics = ["cpu_percent", "memory_percent", "websocket_message_rate", 
                      "websocket_response_time", "websocket_error_rate"]
        
        for metric in key_metrics:
            avg_value = self.tracker.get_average(metric, 300)  # 5 minutes
            if avg_value is not None:
                summary["metrics"][metric] = {
                    "current": avg_value,
                    "trend": self.analyze_trends(metric, 1800)["trend"]  # 30 minutes
                }
        
        # Active bottlenecks
        active_bottlenecks = self.tracker.get_active_bottlenecks()
        summary["bottlenecks"] = [asdict(b) for b in active_bottlenecks]
        
        # Calculate health score
        health_penalties = 0
        for bottleneck in active_bottlenecks:
            if bottleneck.severity == "critical":
                health_penalties += 30
            elif bottleneck.severity == "high":
                health_penalties += 20
            elif bottleneck.severity == "medium":
                health_penalties += 10
            else:
                health_penalties += 5
        
        summary["health_score"] = max(0, 100 - health_penalties)
        
        # Generate recommendations
        if summary["health_score"] < 80:
            summary["recommendations"].append("System performance is degraded. Review active bottlenecks.")
        
        cpu_avg = summary["metrics"].get("cpu_percent", {}).get("current")
        if cpu_avg and cpu_avg > 70:
            summary["recommendations"].append("High CPU usage detected. Consider scaling or optimization.")
        
        memory_avg = summary["metrics"].get("memory_percent", {}).get("current")
        if memory_avg and memory_avg > 80:
            summary["recommendations"].append("High memory usage detected. Monitor for memory leaks.")
        
        return summary
    
    def export_performance_report(self, time_range_seconds: int = 3600) -> str:
        """Export detailed performance report as JSON"""
        report = {
            "generated_at": time.time(),
            "time_range_seconds": time_range_seconds,
            "summary": self.get_performance_summary(),
            "detailed_metrics": {},
            "bottleneck_history": list(self.tracker.bottlenecks.values())
        }
        
        # Include detailed metrics
        all_metric_names = set()
        for metric_name in self.tracker.metrics.keys():
            all_metric_names.add(metric_name)
        
        for metric_name in all_metric_names:
            metrics = self.tracker.get_metrics(metric_name, time_range_seconds)
            if metrics:
                report["detailed_metrics"][metric_name] = {
                    "data_points": [asdict(m) for m in metrics],
                    "analysis": self.analyze_trends(metric_name, time_range_seconds)
                }
        
        return json.dumps(report, indent=2, default=str)

# Global instances
performance_tracker = PerformanceTracker()
system_monitor = SystemMonitor(performance_tracker)
websocket_monitor = WebSocketPerformanceMonitor(performance_tracker)
performance_analyzer = PerformanceAnalyzer(performance_tracker)