# Real-time Collaboration System for IPO Valuation Platform

A comprehensive real-time collaboration system built for the Uprez IPO valuation platform, featuring WebSocket architecture, operational transforms, live data streaming, and advanced collaboration features.

## 🚀 Features

### Core Real-time Infrastructure
- **WebSocket Management**: Scalable WebSocket connections with auto-reconnection and load balancing
- **Redis Pub/Sub**: Horizontal scaling support with message distribution across multiple servers
- **Connection Pooling**: Efficient connection management with configurable limits and timeouts
- **Message Queuing**: Reliable message delivery with batching and compression

### Collaborative Editing
- **Operational Transforms**: Advanced conflict resolution for simultaneous document edits
- **Real-time Synchronization**: Instant document updates across all connected users
- **Multi-user Cursors**: Live cursor tracking and presence indicators
- **Version Control**: Complete edit history with branching and merging capabilities
- **Undo/Redo**: Cross-user undo/redo functionality

### Live Data Streaming
- **Valuation Updates**: Real-time streaming of financial calculations and model updates
- **Market Data**: Live market data integration with automatic refresh
- **Chart Updates**: Dynamic financial chart updates as data changes
- **Calculation Progress**: Real-time progress indicators for complex calculations
- **Error Notifications**: Instant error reporting and status updates

### Collaboration Features
- **Comments & Annotations**: Rich commenting system with replies and resolutions
- **User Presence**: Real-time user presence indicators and activity tracking
- **Activity Feeds**: Comprehensive activity logging and notifications
- **Team Workspaces**: Multi-workspace support with permission controls
- **Document Sharing**: Real-time document sharing and collaboration

### Performance & Scaling
- **Performance Monitoring**: Comprehensive system performance tracking
- **Bottleneck Detection**: Automatic performance issue identification
- **Auto-scaling**: Dynamic scaling based on load and performance metrics
- **Memory Optimization**: Efficient memory usage with automatic cleanup
- **Connection Throttling**: Rate limiting and connection management

## 📁 Project Structure

```
src/realtime/
├── backend/                    # Backend Python modules
│   ├── websocket_manager.py   # WebSocket connection management
│   ├── operational_transform.py # OT engine for collaborative editing
│   ├── live_data_sync.py      # Real-time data streaming
│   ├── collaboration_features.py # Comments, annotations, presence
│   ├── performance_monitoring.py # Performance tracking and optimization
│   └── websocket_endpoints.py # FastAPI WebSocket endpoints
├── frontend/                   # Frontend React components and hooks
│   ├── hooks/                 # React hooks for real-time features
│   │   ├── useWebSocket.ts   # WebSocket connection management
│   │   ├── useCollaboration.ts # Collaboration features hook
│   │   └── useDataStream.ts  # Data streaming hook
│   └── components/           # React components
│       ├── CollaborationPanel.tsx # Comments and collaboration UI
│       └── RealTimeStatusBar.tsx  # Connection status and presence
├── tests/                    # Comprehensive test suite
│   ├── test_websocket_manager.py
│   ├── test_operational_transform.py
│   └── test_collaboration_features.py
├── config/                   # Configuration management
│   └── settings.py          # Centralized settings and environment config
├── examples/                 # Usage examples and integration guides
│   ├── basic_usage.py       # Basic setup and usage examples
│   └── advanced_features.py # Advanced features and use cases
└── docs/                    # Documentation
    └── README.md           # This file
```

## 🚀 Quick Start

### Backend Setup

1. **Install Dependencies**:
```bash
pip install fastapi uvicorn redis websockets psutil
```

2. **Start Redis Server**:
```bash
redis-server
```

3. **Configure Environment Variables**:
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export WS_HOST=0.0.0.0
export WS_PORT=8000
export JWT_SECRET_KEY=your-secret-key
```

4. **Run the Server**:
```python
from src.realtime.examples.basic_usage import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Frontend Setup

1. **Install Dependencies**:
```bash
npm install
```

2. **Use Real-time Hooks**:
```typescript
import { useWebSocket } from './src/realtime/frontend/hooks/useWebSocket';
import { useCollaboration } from './src/realtime/frontend/hooks/useCollaboration';

const MyComponent = () => {
  const webSocket = useWebSocket({
    url: 'ws://localhost:8000',
    token: 'your-jwt-token',
    workspaceId: 'workspace_123',
    userId: 'user_456',
    sessionId: 'session_789'
  });

  const collaboration = useCollaboration(webSocket, {
    workspaceId: 'workspace_123',
    documentId: 'doc_123',
    userId: 'user_456',
    username: 'john_doe',
    displayName: 'John Doe'
  });

  return (
    <div>
      <div>Status: {webSocket.status}</div>
      <div>Comments: {collaboration.comments.length}</div>
    </div>
  );
};
```

## 🔧 Configuration

### WebSocket Configuration

```python
from src.realtime.config.settings import settings

# Access WebSocket settings
websocket_config = settings.websocket
print(f"Max connections per workspace: {websocket_config.max_connections_per_workspace}")
print(f"Heartbeat interval: {websocket_config.heartbeat_interval_seconds}s")
```

### Redis Configuration

```python
# Redis settings
redis_config = settings.redis
print(f"Redis URL: {settings.get_redis_url()}")
```

### Performance Configuration

```python
# Performance monitoring
perf_config = settings.performance
print(f"CPU threshold: {perf_config.cpu_threshold_percent}%")
print(f"Auto-scaling enabled: {perf_config.auto_scaling_enabled}")
```

## 📊 Performance Monitoring

### Real-time Metrics

The system provides comprehensive performance monitoring:

```python
from src.realtime.backend.performance_monitoring import performance_analyzer

# Get performance summary
summary = performance_analyzer.get_performance_summary()
print(f"System health score: {summary['health_score']}")

# Export detailed report
report = performance_analyzer.export_performance_report(3600)  # Last hour
```

### Bottleneck Detection

Automatic bottleneck detection with suggested solutions:

```python
from src.realtime.backend.performance_monitoring import performance_tracker

# Get active bottlenecks
bottlenecks = performance_tracker.get_active_bottlenecks()
for bottleneck in bottlenecks:
    print(f"Issue: {bottleneck.description}")
    print(f"Severity: {bottleneck.severity}")
    print(f"Suggested actions: {bottleneck.suggested_actions}")
```

## 🔄 Operational Transforms

### Document Collaboration

Real-time collaborative editing with conflict resolution:

```python
from src.realtime.backend.operational_transform import ot_engine, Operation, OperationType, Delta

# Create document
document = ot_engine.create_document("doc_123", "Hello World")

# Create operation
operation = Operation(
    type=OperationType.INSERT,
    position=6,
    content=" Beautiful",
    user_id="user_123"
)

# Apply operation
delta = Delta(
    operations=[operation],
    base_version=document.version,
    result_version=document.version + 1,
    document_id="doc_123",
    user_id="user_123",
    session_id="session_123",
    timestamp=time.time()
)

success, error, updated_doc = ot_engine.apply_delta("doc_123", delta)
print(f"Updated content: {updated_doc.content}")  # "Hello Beautiful World"
```

## 💬 Collaboration Features

### Comments and Annotations

```python
from src.realtime.backend.collaboration_features import collaboration_manager

# Add comment
comment = collaboration_manager.add_comment(
    document_id="doc_123",
    workspace_id="workspace_123",
    user_id="user_123",
    content="This section needs review",
    position={"line": 10, "column": 5}
)

# Reply to comment
reply = collaboration_manager.reply_to_comment(
    comment_id=comment.comment_id,
    user_id="user_456",
    content="I agree, let's update it"
)

# Add annotation
annotation = collaboration_manager.add_annotation(
    document_id="doc_123",
    workspace_id="workspace_123",
    user_id="user_123",
    annotation_type="highlight",
    content="Important calculation",
    position={"start": 100, "end": 150},
    style={"backgroundColor": "yellow"}
)
```

### User Presence

```python
# Update user presence
presence = collaboration_manager.update_user_presence(
    user_id="user_123",
    workspace_id="workspace_123",
    document_id="doc_123",
    status="online",
    cursor_position={"position": 42, "timestamp": time.time()},
    activity="editing"
)

# Get workspace presence
users = collaboration_manager.get_workspace_presence("workspace_123")
print(f"Active users: {len(users)}")
```

## 📈 Data Streaming

### Real-time Valuation Updates

```python
from src.realtime.backend.live_data_sync import valuation_stream_handler

# Stream valuation update
await valuation_stream_handler.stream_valuation_update(
    workspace_id="workspace_123",
    document_id="valuation_model",
    valuation_data={
        "enterprise_value": 1500000000,
        "equity_value": 1200000000,
        "price_per_share": 24.50,
        "methodology": "DCF"
    },
    user_id="analyst_123"
)

# Stream calculation progress
await valuation_stream_handler.stream_calculation_progress(
    workspace_id="workspace_123",
    document_id="valuation_model",
    progress_data={
        "calculation_id": "dcf_calc_123",
        "progress": 75,
        "status": "Calculating terminal value...",
        "elapsed_time": 45.2
    }
)
```

### Market Data Streaming

```python
# Stream market data updates
await valuation_stream_handler.stream_market_data_update(
    workspace_id="workspace_123",
    market_data={
        "symbol": "AAPL",
        "price": 150.25,
        "change": 2.34,
        "change_percent": 1.58,
        "volume": 45678901
    }
)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest src/realtime/tests/ -v

# Run specific test file
python -m pytest src/realtime/tests/test_websocket_manager.py -v

# Run with coverage
python -m pytest src/realtime/tests/ --cov=src/realtime/backend --cov-report=html
```

### Test Coverage

The test suite covers:
- ✅ WebSocket connection management
- ✅ Operational transform algorithms
- ✅ Collaboration features (comments, annotations, presence)
- ✅ Live data streaming
- ✅ Performance monitoring
- ✅ Error handling and edge cases

## 🔧 Advanced Features

### Multi-Document Synchronization

Synchronize changes across related documents:

```python
from src.realtime.examples.advanced_features import MultiDocumentSync

sync = MultiDocumentSync()
sync.add_document_relationship("main_doc", ["assumptions_doc", "outputs_doc"])
await sync.sync_document_changes("main_doc", {"discount_rate": 0.10})
```

### Intelligent Conflict Resolution

ML-powered conflict resolution with user priorities:

```python
from src.realtime.examples.advanced_features import IntelligentConflictResolver

resolver = IntelligentConflictResolver()
resolver.set_user_priority("senior_analyst", 3)
resolver.set_content_importance("valuation", 2.0)
resolved_ops = resolver.resolve_conflict([delta1, delta2, delta3])
```

### Real-time Financial Calculator

Stream calculation progress and results:

```python
from src.realtime.examples.advanced_features import RealTimeFinancialCalculator

calculator = RealTimeFinancialCalculator()
await calculator.start()

await calculator.queue_calculation("dcf_valuation", "dcf_valuation", {
    "revenue": 150000000,
    "growth_rate": 0.08,
    "discount_rate": 0.12
}, "workspace_123")
```

## 🌐 Production Deployment

### Docker Setup

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
EXPOSE 8000

CMD ["uvicorn", "src.realtime.examples.basic_usage:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Production environment variables
ENVIRONMENT=production
DATABASE_URL=postgresql://user:pass@db:5432/uprez_prod
REDIS_HOST=redis
REDIS_PORT=6379
JWT_SECRET_KEY=your-production-secret
WS_MAX_CONNECTIONS_PER_WORKSPACE=500
PERF_METRICS_ENABLED=true
AUTO_SCALING_ENABLED=true
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realtime-collaboration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: realtime-collaboration
  template:
    metadata:
      labels:
        app: realtime-collaboration
    spec:
      containers:
      - name: realtime-collaboration
        image: uprez/realtime-collaboration:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
```

## 🔒 Security

### JWT Authentication

```python
from src.realtime.config.settings import settings

# JWT configuration
jwt_config = settings.security
print(f"Algorithm: {jwt_config.jwt_algorithm}")
print(f"Expiration: {jwt_config.jwt_expiration_hours} hours")
```

### Rate Limiting

```python
# Rate limiting configuration
rate_limit = settings.security.rate_limit_requests_per_minute
print(f"Rate limit: {rate_limit} requests per minute")
```

### Input Sanitization

```python
# Input validation
max_length = settings.security.max_input_length
sanitize = settings.security.sanitize_input
```

## 📚 API Reference

### WebSocket Message Types

#### Document Operations
- `document_edit`: Edit document content
- `cursor_update`: Update cursor position
- `selection_update`: Update text selection
- `document_updated`: Document content changed

#### Collaboration
- `comment_add`: Add comment
- `comment_reply`: Reply to comment
- `comment_resolved`: Comment resolved
- `annotation_create`: Create annotation
- `presence_update`: User presence changed

#### Data Streaming
- `subscribe_stream`: Subscribe to data streams
- `stream_data`: Streaming data update
- `valuation_update`: Valuation calculation result
- `market_data`: Market data update
- `calculation_progress`: Calculation progress

### REST API Endpoints

#### Status and Management
- `GET /api/workspace/{workspace_id}/status`: Get workspace status
- `GET /api/document/{document_id}/collaboration`: Get collaboration info
- `GET /api/document/{document_id}/comments`: Get document comments
- `GET /api/document/{document_id}/annotations`: Get document annotations
- `GET /api/system/stats`: Get system statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

This project is part of the Uprez IPO valuation platform. All rights reserved.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation and examples

## 🚀 Roadmap

### Upcoming Features
- [ ] Voice collaboration with WebRTC
- [ ] Advanced ML-powered conflict resolution
- [ ] Mobile app integration
- [ ] Blockchain-based audit trails
- [ ] Advanced analytics dashboard
- [ ] Integration with external data providers
- [ ] Multi-language support
- [ ] Advanced permission system

### Performance Improvements
- [ ] WebAssembly for client-side OT
- [ ] GPU-accelerated calculations
- [ ] Edge caching for better latency
- [ ] Advanced compression algorithms

---

Built with ❤️ for the Uprez IPO valuation platform.