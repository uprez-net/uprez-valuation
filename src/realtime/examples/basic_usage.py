"""
Basic Usage Examples for Real-time Collaboration System
Demonstrates how to set up and use the real-time collaboration features
"""

import asyncio
import json
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import our real-time collaboration components
from ..backend.websocket_manager import websocket_manager
from ..backend.operational_transform import ot_engine
from ..backend.live_data_sync import data_stream_manager, valuation_stream_handler
from ..backend.collaboration_features import collaboration_manager
from ..backend.performance_monitoring import system_monitor, performance_tracker
from ..config.settings import settings

# Example 1: Basic FastAPI Server Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await websocket_manager.start()
    data_stream_manager.start()
    await system_monitor.start()
    
    # Create sample document
    ot_engine.create_document("sample_doc", "Hello World!", {"title": "Sample Document"})
    
    yield
    
    # Shutdown
    await websocket_manager.stop()
    data_stream_manager.stop()
    await system_monitor.stop()

app = FastAPI(lifespan=lifespan, title="Real-time Collaboration Server")

# Add the WebSocket endpoint
@app.websocket("/ws/{workspace_id}")
async def websocket_endpoint(websocket, workspace_id: str, token: str, session_id: str, user_id: str = None):
    """Main WebSocket endpoint"""
    from ..backend.websocket_endpoints import websocket_endpoint as ws_handler
    await ws_handler(websocket, workspace_id, token, session_id, user_id)

# Example 2: Document Collaboration Setup
async def setup_document_collaboration_example():
    """Example of setting up document collaboration"""
    
    # Create a document
    document_id = "valuation_model_v1"
    document = ot_engine.create_document(
        document_id=document_id,
        initial_content="""
        # IPO Valuation Model
        
        ## Company Information
        Company: Example Corp
        Industry: Technology
        Revenue: $100M
        
        ## Valuation Methods
        1. DCF Analysis
        2. Comparable Company Analysis
        3. Precedent Transaction Analysis
        """,
        metadata={
            "title": "IPO Valuation Model v1",
            "created_by": "analyst1",
            "workspace_id": "workspace_123"
        }
    )
    
    print(f"Created document: {document.document_id}")
    print(f"Initial content length: {len(document.content)} characters")
    return document

# Example 3: Real-time Data Streaming Setup
async def setup_data_streaming_example():
    """Example of setting up real-time data streaming"""
    
    workspace_id = "workspace_123"
    
    # Simulate valuation update
    valuation_data = {
        "enterprise_value": 1500000000,  # $1.5B
        "equity_value": 1200000000,      # $1.2B
        "price_per_share": 24.50,
        "market_cap": 1200000000,
        "methodology": "DCF",
        "confidence_level": 0.85
    }
    
    await valuation_stream_handler.stream_valuation_update(
        workspace_id=workspace_id,
        document_id="valuation_model_v1",
        valuation_data=valuation_data,
        user_id="analyst1"
    )
    
    # Simulate market data update
    market_data = {
        "symbol": "COMP",
        "price": 45.67,
        "change": 2.34,
        "change_percent": 5.4,
        "volume": 1234567,
        "market_cap": 2300000000
    }
    
    await valuation_stream_handler.stream_market_data_update(
        workspace_id=workspace_id,
        market_data=market_data
    )
    
    print("Streamed valuation and market data updates")

# Example 4: Operational Transform Usage
async def operational_transform_example():
    """Example of using operational transforms for collaborative editing"""
    
    from ..backend.operational_transform import Operation, OperationType, Delta
    import time
    
    document_id = "collaborative_doc"
    
    # Create document
    ot_engine.create_document(document_id, "The quick brown fox jumps")
    
    # User 1 makes an edit (insert " over the lazy dog" at end)
    user1_ops = [Operation(
        type=OperationType.INSERT,
        position=25,  # End of string
        content=" over the lazy dog",
        user_id="user1"
    )]
    
    user1_delta = Delta(
        operations=user1_ops,
        base_version=0,
        result_version=1,
        document_id=document_id,
        user_id="user1",
        session_id="session1",
        timestamp=time.time()
    )
    
    # User 2 makes an edit simultaneously (insert "very " before "quick")
    user2_ops = [Operation(
        type=OperationType.INSERT,
        position=4,   # Before "quick"
        content="very ",
        user_id="user2"
    )]
    
    user2_delta = Delta(
        operations=user2_ops,
        base_version=0,  # Same base version (conflict!)
        result_version=1,
        document_id=document_id,
        user_id="user2",
        session_id="session2",
        timestamp=time.time() + 0.1  # Slightly later
    )
    
    # Apply first delta
    success1, error1, doc1 = ot_engine.apply_delta(document_id, user1_delta)
    print(f"User 1 edit applied: {success1}")
    print(f"Content after user 1: {doc1.content}")
    
    # Apply second delta (will be transformed due to conflict)
    success2, error2, doc2 = ot_engine.apply_delta(document_id, user2_delta)
    print(f"User 2 edit applied: {success2}")
    print(f"Final content: {doc2.content}")
    print(f"Final version: {doc2.version}")
    
    return doc2

# Example 5: Collaboration Features Usage
def collaboration_features_example():
    """Example of using collaboration features"""
    
    workspace_id = "workspace_123"
    document_id = "valuation_model_v1"
    
    # Add users
    from ..backend.collaboration_features import User
    
    user1 = User(
        user_id="analyst1",
        username="jane.analyst",
        display_name="Jane Analyst",
        email="jane@example.com",
        role="analyst"
    )
    
    user2 = User(
        user_id="reviewer1",
        username="bob.reviewer",
        display_name="Bob Reviewer", 
        email="bob@example.com",
        role="reviewer"
    )
    
    collaboration_manager.add_user(user1)
    collaboration_manager.add_user(user2)
    
    # Add a comment
    comment = collaboration_manager.add_comment(
        document_id=document_id,
        workspace_id=workspace_id,
        user_id="analyst1",
        content="The DCF assumptions look conservative. Should we increase the growth rate?",
        position={"line": 15, "column": 1, "section": "dcf_analysis"}
    )
    
    # Reply to the comment
    reply = collaboration_manager.reply_to_comment(
        comment_id=comment.comment_id,
        user_id="reviewer1",
        content="Good point. Let's increase it to 8% based on market research."
    )
    
    # Add an annotation
    annotation = collaboration_manager.add_annotation(
        document_id=document_id,
        workspace_id=workspace_id,
        user_id="reviewer1",
        annotation_type="highlight",
        content="Check this calculation",
        position={"start": 450, "end": 500},
        style={"backgroundColor": "yellow", "color": "black"}
    )
    
    # Update user presence
    collaboration_manager.update_user_presence(
        user_id="analyst1",
        workspace_id=workspace_id,
        document_id=document_id,
        status="online",
        cursor_position={"position": 234, "timestamp": time.time()},
        activity="editing"
    )
    
    print(f"Added comment: {comment.comment_id}")
    print(f"Added reply: {reply.reply_id}")
    print(f"Added annotation: {annotation.annotation_id}")
    
    # Get collaboration stats
    stats = collaboration_manager.get_workspace_stats(workspace_id)
    print(f"Workspace stats: {stats}")
    
    return comment, reply, annotation

# Example 6: Performance Monitoring Setup
async def performance_monitoring_example():
    """Example of setting up performance monitoring"""
    
    # Record custom metrics
    performance_tracker.record_metric("custom_calculation_time", 150.5, "valuation_engine")
    performance_tracker.record_metric("document_load_time", 45.2, "document_service")
    performance_tracker.record_metric("websocket_connection_count", 25, "websocket")
    
    # Get performance analysis
    from ..backend.performance_monitoring import performance_analyzer
    
    # Analyze trends
    cpu_trend = performance_analyzer.analyze_trends("cpu_percent", 1800)  # 30 minutes
    print(f"CPU trend analysis: {cpu_trend}")
    
    # Get performance summary
    summary = performance_analyzer.get_performance_summary()
    print(f"Performance health score: {summary['health_score']}")
    
    # Export detailed report
    report = performance_analyzer.export_performance_report(3600)  # 1 hour
    
    # Save report to file
    with open("performance_report.json", "w") as f:
        f.write(report)
    
    print("Performance report saved to performance_report.json")

# Example 7: Complete Integration Example
async def complete_integration_example():
    """Complete example showing all features working together"""
    
    print("=== Real-time Collaboration System Integration Example ===")
    
    # 1. Set up document collaboration
    print("\n1. Setting up document collaboration...")
    document = await setup_document_collaboration_example()
    
    # 2. Set up data streaming
    print("\n2. Setting up data streaming...")
    await setup_data_streaming_example()
    
    # 3. Demonstrate operational transforms
    print("\n3. Demonstrating operational transforms...")
    final_doc = await operational_transform_example()
    
    # 4. Use collaboration features
    print("\n4. Using collaboration features...")
    comment, reply, annotation = collaboration_features_example()
    
    # 5. Monitor performance
    print("\n5. Monitoring performance...")
    await performance_monitoring_example()
    
    print("\n=== Integration Example Complete ===")
    
    return {
        "document": final_doc,
        "comment": comment,
        "reply": reply,
        "annotation": annotation
    }

# Example 8: Client-Side Integration (React)
def react_integration_example():
    """Example of React client-side integration"""
    
    react_code = '''
import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { useCollaboration } from '../hooks/useCollaboration';
import { useDataStream } from '../hooks/useDataStream';
import CollaborationPanel from '../components/CollaborationPanel';
import RealTimeStatusBar from '../components/RealTimeStatusBar';

const ValuationEditor = ({ workspaceId, documentId, userId, token }) => {
  const [content, setContent] = useState('');
  
  // WebSocket connection
  const webSocket = useWebSocket({
    url: 'ws://localhost:8000',
    token,
    workspaceId,
    userId,
    sessionId: `session_${Date.now()}`
  });
  
  // Collaboration features
  const collaboration = useCollaboration(webSocket, {
    workspaceId,
    documentId,
    userId,
    username: 'current_user',
    displayName: 'Current User'
  });
  
  // Data streaming
  const dataStream = useDataStream(webSocket, {
    workspaceId,
    documentId,
    defaultStreamTypes: ['valuation_update', 'calculation_progress']
  });
  
  // Handle document updates
  useEffect(() => {
    const unsubscribe = webSocket.subscribe('document_updated', (message) => {
      const { content: newContent } = message.payload;
      setContent(newContent);
    });
    
    return unsubscribe;
  }, [webSocket]);
  
  // Handle cursor/selection changes
  const handleCursorChange = (position) => {
    collaboration.updateCursor(position);
  };
  
  const handleSelectionChange = (start, end) => {
    collaboration.updateSelection(start, end);
  };
  
  return (
    <div className="valuation-editor">
      <RealTimeStatusBar
        connectionStatus={webSocket.status}
        isConnected={webSocket.isConnected}
        presenceUsers={collaboration.presenceUsers}
        currentUserId={userId}
        connectionQuality={dataStream.connectionQuality}
        messagesReceived={webSocket.connectionStats.messagesReceived}
        messagesSent={webSocket.connectionStats.messagesent}
        onReconnect={webSocket.reconnect}
      />
      
      <div className="editor-container">
        <div className="editor-main">
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onSelect={(e) => handleSelectionChange(e.target.selectionStart, e.target.selectionEnd)}
            className="document-editor"
          />
          
          {/* Live valuation updates */}
          {dataStream.latestData.valuation_update && (
            <div className="valuation-update">
              <h4>Latest Valuation</h4>
              <p>Enterprise Value: ${dataStream.latestData.valuation_update.data.valuation.enterprise_value.toLocaleString()}</p>
              <p>Price per Share: ${dataStream.latestData.valuation_update.data.valuation.price_per_share}</p>
            </div>
          )}
          
          {/* Calculation progress */}
          {dataStream.calculationProgress.length > 0 && (
            <div className="calculation-progress">
              {dataStream.calculationProgress.map(calc => (
                <div key={calc.calculation_id} className="progress-item">
                  <span>{calc.calculation_type}: {calc.progress}%</span>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${calc.progress}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        <CollaborationPanel
          comments={collaboration.comments}
          annotations={collaboration.annotations}
          presenceUsers={collaboration.presenceUsers}
          activities={collaboration.activities}
          currentUserId={userId}
          onAddComment={collaboration.addComment}
          onReplyToComment={collaboration.replyToComment}
          onResolveComment={collaboration.resolveComment}
          onAddAnnotation={collaboration.addAnnotation}
          onDeleteAnnotation={collaboration.deleteAnnotation}
        />
      </div>
    </div>
  );
};

export default ValuationEditor;
    '''
    
    print("React Integration Example:")
    print(react_code)
    
    return react_code

if __name__ == "__main__":
    # Run the complete integration example
    asyncio.run(complete_integration_example())
    
    # Show React integration
    react_integration_example()