"""
Advanced Features Examples
Demonstrates advanced real-time collaboration features and use cases
"""

import asyncio
import json
import time
from typing import Dict, List, Any
from dataclasses import asdict

from ..backend.websocket_manager import websocket_manager, Message
from ..backend.operational_transform import ot_engine, Operation, OperationType, Delta
from ..backend.live_data_sync import data_stream_manager, StreamType, StreamData
from ..backend.collaboration_features import collaboration_manager
from ..backend.performance_monitoring import performance_tracker, system_monitor
from ..config.settings import settings

# Advanced Example 1: Multi-Document Synchronization
class MultiDocumentSync:
    """Handles synchronization across multiple related documents"""
    
    def __init__(self):
        self.document_relationships: Dict[str, List[str]] = {}
        self.sync_rules: Dict[str, Dict[str, Any]] = {}
    
    def add_document_relationship(self, primary_doc: str, related_docs: List[str], 
                                 sync_rule: str = "bidirectional"):
        """Add relationship between documents"""
        self.document_relationships[primary_doc] = related_docs
        self.sync_rules[primary_doc] = {
            "rule": sync_rule,
            "fields": [],
            "auto_sync": True
        }
    
    async def sync_document_changes(self, document_id: str, changes: Dict[str, Any]):
        """Synchronize changes across related documents"""
        if document_id not in self.document_relationships:
            return
        
        related_docs = self.document_relationships[document_id]
        sync_rule = self.sync_rules[document_id]
        
        for related_doc_id in related_docs:
            await self._apply_sync_changes(related_doc_id, changes, sync_rule)
    
    async def _apply_sync_changes(self, target_doc: str, changes: Dict[str, Any], 
                                 sync_rule: Dict[str, Any]):
        """Apply synchronized changes to target document"""
        # Get target document
        target_document = ot_engine.get_document(target_doc)
        if not target_document:
            return
        
        # Create operations based on sync rule
        operations = []
        
        if sync_rule["rule"] == "bidirectional":
            # Apply all changes
            for field, value in changes.items():
                # Create operation to update field
                op = Operation(
                    type=OperationType.INSERT,
                    position=0,  # Simplified - would need proper positioning
                    content=f"<!-- Synced update: {field} = {value} -->\n",
                    user_id="system"
                )
                operations.append(op)
        
        if operations:
            delta = Delta(
                operations=operations,
                base_version=target_document.version,
                result_version=target_document.version + 1,
                document_id=target_doc,
                user_id="system",
                session_id="sync",
                timestamp=time.time()
            )
            
            success, error, updated_doc = ot_engine.apply_delta(target_doc, delta)
            if success:
                # Notify connected users
                await self._notify_sync_update(target_doc, changes)
    
    async def _notify_sync_update(self, document_id: str, changes: Dict[str, Any]):
        """Notify users of synchronized updates"""
        # Find workspace for document
        doc = ot_engine.get_document(document_id)
        if not doc or "workspace_id" not in doc.metadata:
            return
        
        workspace_id = doc.metadata["workspace_id"]
        
        message = Message(
            type="document_sync_update",
            payload={
                "document_id": document_id,
                "changes": changes,
                "source": "multi_document_sync"
            },
            user_id="system",
            session_id="sync",
            workspace_id=workspace_id,
            timestamp=time.time()
        )
        
        await websocket_manager.broadcast_to_workspace(workspace_id, message)

# Advanced Example 2: Intelligent Conflict Resolution
class IntelligentConflictResolver:
    """Advanced conflict resolution using ML-like heuristics"""
    
    def __init__(self):
        self.user_priorities: Dict[str, int] = {}
        self.operation_weights: Dict[str, float] = {
            "insert": 1.0,
            "delete": 0.8,
            "format": 0.5
        }
        self.content_importance: Dict[str, float] = {}
    
    def set_user_priority(self, user_id: str, priority: int):
        """Set priority for user (higher number = higher priority)"""
        self.user_priorities[user_id] = priority
    
    def set_content_importance(self, content_pattern: str, importance: float):
        """Set importance weight for content patterns"""
        self.content_importance[content_pattern] = importance
    
    def resolve_conflict(self, operations: List[Delta]) -> List[Delta]:
        """Resolve conflicts between multiple operations intelligently"""
        if len(operations) <= 1:
            return operations
        
        # Score each operation
        scored_operations = []
        for delta in operations:
            score = self._calculate_operation_score(delta)
            scored_operations.append((score, delta))
        
        # Sort by score (highest first)
        scored_operations.sort(key=lambda x: x[0], reverse=True)
        
        # Apply operations in priority order with transformations
        resolved_operations = []
        for _, delta in scored_operations:
            # Transform against previously accepted operations
            transformed_delta = self._transform_against_accepted(delta, resolved_operations)
            resolved_operations.append(transformed_delta)
        
        return resolved_operations
    
    def _calculate_operation_score(self, delta: Delta) -> float:
        """Calculate priority score for an operation"""
        score = 0.0
        
        # User priority weight
        user_priority = self.user_priorities.get(delta.user_id, 1)
        score += user_priority * 10
        
        # Operation type weight
        for op in delta.operations:
            op_weight = self.operation_weights.get(op.type.value, 1.0)
            score += op_weight
            
            # Content importance
            for pattern, importance in self.content_importance.items():
                if pattern in op.content:
                    score += importance * 5
        
        # Recency boost (more recent = higher score)
        time_diff = time.time() - delta.timestamp
        recency_boost = max(0, 10 - (time_diff / 60))  # 10 points max, decays over 10 minutes
        score += recency_boost
        
        return score
    
    def _transform_against_accepted(self, delta: Delta, accepted: List[Delta]) -> Delta:
        """Transform delta against accepted operations"""
        if not accepted:
            return delta
        
        transformed_ops = delta.operations[:]
        
        for accepted_delta in accepted:
            for op in transformed_ops:
                for accepted_op in accepted_delta.operations:
                    # Apply transformation logic
                    op = ot_engine._transform_operation(op, accepted_op)
            
        return Delta(
            operations=transformed_ops,
            base_version=delta.base_version,
            result_version=delta.result_version,
            document_id=delta.document_id,
            user_id=delta.user_id,
            session_id=delta.session_id,
            timestamp=delta.timestamp
        )

# Advanced Example 3: Real-time Financial Model Calculator
class RealTimeFinancialCalculator:
    """Real-time financial calculations with streaming results"""
    
    def __init__(self):
        self.calculation_dependencies: Dict[str, List[str]] = {}
        self.cached_results: Dict[str, Any] = {}
        self.calculation_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    async def start(self):
        """Start the calculation engine"""
        self._running = True
        asyncio.create_task(self._calculation_worker())
    
    async def stop(self):
        """Stop the calculation engine"""
        self._running = False
    
    def add_calculation_dependency(self, calc_id: str, dependencies: List[str]):
        """Add dependencies between calculations"""
        self.calculation_dependencies[calc_id] = dependencies
    
    async def queue_calculation(self, calc_id: str, calc_type: str, 
                              parameters: Dict[str, Any], workspace_id: str):
        """Queue a calculation for processing"""
        await self.calculation_queue.put({
            "calc_id": calc_id,
            "calc_type": calc_type,
            "parameters": parameters,
            "workspace_id": workspace_id,
            "timestamp": time.time()
        })
    
    async def _calculation_worker(self):
        """Background worker for processing calculations"""
        while self._running:
            try:
                calc_task = await asyncio.wait_for(self.calculation_queue.get(), timeout=1.0)
                await self._process_calculation(calc_task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in calculation worker: {e}")
    
    async def _process_calculation(self, calc_task: Dict[str, Any]):
        """Process a single calculation"""
        calc_id = calc_task["calc_id"]
        calc_type = calc_task["calc_type"]
        parameters = calc_task["parameters"]
        workspace_id = calc_task["workspace_id"]
        
        # Check dependencies
        dependencies = self.calculation_dependencies.get(calc_id, [])
        for dep_id in dependencies:
            if dep_id not in self.cached_results:
                # Dependency not ready, requeue
                await asyncio.sleep(0.1)
                await self.calculation_queue.put(calc_task)
                return
        
        # Stream calculation start
        await self._stream_calculation_progress(workspace_id, calc_id, 0, "Starting calculation...")
        
        # Perform calculation based on type
        result = None
        if calc_type == "dcf_valuation":
            result = await self._calculate_dcf(parameters, workspace_id, calc_id)
        elif calc_type == "comparable_analysis":
            result = await self._calculate_comparables(parameters, workspace_id, calc_id)
        elif calc_type == "precedent_transactions":
            result = await self._calculate_precedents(parameters, workspace_id, calc_id)
        elif calc_type == "weighted_average":
            result = await self._calculate_weighted_average(parameters, workspace_id, calc_id)
        
        # Cache result
        if result:
            self.cached_results[calc_id] = result
            
            # Stream final result
            await self._stream_calculation_complete(workspace_id, calc_id, result)
            
            # Trigger dependent calculations
            await self._trigger_dependent_calculations(calc_id, workspace_id)
    
    async def _calculate_dcf(self, params: Dict[str, Any], workspace_id: str, calc_id: str) -> Dict[str, Any]:
        """Calculate DCF valuation with streaming progress"""
        
        # Extract parameters
        revenue = params.get("revenue", 100000000)
        growth_rate = params.get("growth_rate", 0.05)
        discount_rate = params.get("discount_rate", 0.10)
        terminal_growth = params.get("terminal_growth", 0.02)
        projection_years = params.get("projection_years", 5)
        
        await self._stream_calculation_progress(workspace_id, calc_id, 20, "Calculating revenue projections...")
        await asyncio.sleep(0.5)  # Simulate computation
        
        # Project revenues
        revenues = []
        current_revenue = revenue
        for year in range(projection_years):
            current_revenue *= (1 + growth_rate)
            revenues.append(current_revenue)
        
        await self._stream_calculation_progress(workspace_id, calc_id, 40, "Calculating free cash flows...")
        await asyncio.sleep(0.5)
        
        # Simplified FCF calculation (assume 20% FCF margin)
        free_cash_flows = [rev * 0.20 for rev in revenues]
        
        await self._stream_calculation_progress(workspace_id, calc_id, 60, "Discounting cash flows...")
        await asyncio.sleep(0.5)
        
        # Discount cash flows
        discounted_fcfs = []
        for i, fcf in enumerate(free_cash_flows):
            discounted_fcf = fcf / ((1 + discount_rate) ** (i + 1))
            discounted_fcfs.append(discounted_fcf)
        
        await self._stream_calculation_progress(workspace_id, calc_id, 80, "Calculating terminal value...")
        await asyncio.sleep(0.5)
        
        # Terminal value
        terminal_fcf = free_cash_flows[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        discounted_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)
        
        # Enterprise value
        enterprise_value = sum(discounted_fcfs) + discounted_terminal_value
        
        await self._stream_calculation_progress(workspace_id, calc_id, 100, "DCF calculation complete")
        
        return {
            "method": "dcf",
            "enterprise_value": enterprise_value,
            "revenues": revenues,
            "free_cash_flows": free_cash_flows,
            "discounted_fcfs": discounted_fcfs,
            "terminal_value": terminal_value,
            "discount_rate": discount_rate,
            "growth_rate": growth_rate,
            "assumptions": params
        }
    
    async def _calculate_comparables(self, params: Dict[str, Any], workspace_id: str, calc_id: str) -> Dict[str, Any]:
        """Calculate comparable company analysis"""
        
        comparables = params.get("comparables", [])
        target_metric = params.get("target_metric", "revenue")
        target_value = params.get("target_value", 100000000)
        
        await self._stream_calculation_progress(workspace_id, calc_id, 25, "Analyzing comparable companies...")
        await asyncio.sleep(0.3)
        
        # Calculate multiples
        multiples = []
        for comp in comparables:
            if target_metric in comp and "enterprise_value" in comp:
                multiple = comp["enterprise_value"] / comp[target_metric]
                multiples.append(multiple)
        
        await self._stream_calculation_progress(workspace_id, calc_id, 50, "Calculating valuation multiples...")
        await asyncio.sleep(0.3)
        
        if multiples:
            median_multiple = sorted(multiples)[len(multiples) // 2]
            mean_multiple = sum(multiples) / len(multiples)
            
            # Apply to target
            implied_value_median = target_value * median_multiple
            implied_value_mean = target_value * mean_multiple
        else:
            median_multiple = mean_multiple = 0
            implied_value_median = implied_value_mean = 0
        
        await self._stream_calculation_progress(workspace_id, calc_id, 100, "Comparable analysis complete")
        
        return {
            "method": "comparables",
            "enterprise_value": implied_value_median,
            "multiples": multiples,
            "median_multiple": median_multiple,
            "mean_multiple": mean_multiple,
            "implied_value_median": implied_value_median,
            "implied_value_mean": implied_value_mean,
            "comparables_count": len(comparables)
        }
    
    async def _calculate_precedents(self, params: Dict[str, Any], workspace_id: str, calc_id: str) -> Dict[str, Any]:
        """Calculate precedent transaction analysis"""
        
        transactions = params.get("transactions", [])
        target_metric = params.get("target_metric", "revenue")
        target_value = params.get("target_value", 100000000)
        
        await self._stream_calculation_progress(workspace_id, calc_id, 30, "Analyzing precedent transactions...")
        await asyncio.sleep(0.3)
        
        # Calculate transaction multiples
        multiples = []
        for txn in transactions:
            if target_metric in txn and "transaction_value" in txn:
                multiple = txn["transaction_value"] / txn[target_metric]
                multiples.append(multiple)
        
        await self._stream_calculation_progress(workspace_id, calc_id, 70, "Applying transaction multiples...")
        await asyncio.sleep(0.3)
        
        if multiples:
            median_multiple = sorted(multiples)[len(multiples) // 2]
            implied_value = target_value * median_multiple
        else:
            median_multiple = 0
            implied_value = 0
        
        await self._stream_calculation_progress(workspace_id, calc_id, 100, "Precedent analysis complete")
        
        return {
            "method": "precedents",
            "enterprise_value": implied_value,
            "transaction_multiples": multiples,
            "median_multiple": median_multiple,
            "transactions_count": len(transactions)
        }
    
    async def _calculate_weighted_average(self, params: Dict[str, Any], workspace_id: str, calc_id: str) -> Dict[str, Any]:
        """Calculate weighted average of all methods"""
        
        weights = params.get("weights", {})
        
        await self._stream_calculation_progress(workspace_id, calc_id, 50, "Calculating weighted average...")
        await asyncio.sleep(0.2)
        
        total_value = 0
        total_weight = 0
        
        for method, weight in weights.items():
            if method in self.cached_results:
                method_value = self.cached_results[method]["enterprise_value"]
                total_value += method_value * weight
                total_weight += weight
        
        weighted_value = total_value / total_weight if total_weight > 0 else 0
        
        await self._stream_calculation_progress(workspace_id, calc_id, 100, "Weighted average complete")
        
        return {
            "method": "weighted_average",
            "enterprise_value": weighted_value,
            "weights": weights,
            "total_weight": total_weight
        }
    
    async def _stream_calculation_progress(self, workspace_id: str, calc_id: str, 
                                        progress: float, status: str):
        """Stream calculation progress updates"""
        from ..backend.live_data_sync import financial_metrics_streamer
        
        await financial_metrics_streamer.update_calculation_progress(
            calculation_id=calc_id,
            progress=progress,
            status_message=status
        )
    
    async def _stream_calculation_complete(self, workspace_id: str, calc_id: str, result: Dict[str, Any]):
        """Stream calculation completion"""
        from ..backend.live_data_sync import financial_metrics_streamer
        
        await financial_metrics_streamer.complete_calculation(
            calculation_id=calc_id,
            final_results=result
        )
    
    async def _trigger_dependent_calculations(self, completed_calc_id: str, workspace_id: str):
        """Trigger calculations that depend on the completed one"""
        for calc_id, dependencies in self.calculation_dependencies.items():
            if completed_calc_id in dependencies:
                # Check if all dependencies are met
                all_deps_met = all(dep in self.cached_results for dep in dependencies)
                if all_deps_met and calc_id not in self.cached_results:
                    # This is a placeholder - in real implementation, you'd have
                    # the original calculation parameters stored somewhere
                    print(f"Would trigger dependent calculation: {calc_id}")

# Advanced Example 4: Usage of Advanced Features
async def advanced_features_example():
    """Demonstrate advanced collaboration features"""
    
    print("=== Advanced Real-time Collaboration Features ===")
    
    # 1. Multi-Document Synchronization
    print("\n1. Setting up multi-document synchronization...")
    multi_sync = MultiDocumentSync()
    
    # Create related documents
    main_doc = ot_engine.create_document("valuation_main", "# Main Valuation Model", 
                                       {"workspace_id": "workspace_123"})
    assumptions_doc = ot_engine.create_document("valuation_assumptions", "# Assumptions", 
                                              {"workspace_id": "workspace_123"})
    outputs_doc = ot_engine.create_document("valuation_outputs", "# Outputs", 
                                          {"workspace_id": "workspace_123"})
    
    # Set up relationships
    multi_sync.add_document_relationship("valuation_main", 
                                       ["valuation_assumptions", "valuation_outputs"])
    
    # Simulate synchronized change
    await multi_sync.sync_document_changes("valuation_main", 
                                         {"discount_rate": 0.10, "growth_rate": 0.05})
    
    # 2. Intelligent Conflict Resolution
    print("\n2. Setting up intelligent conflict resolution...")
    resolver = IntelligentConflictResolver()
    
    # Set user priorities
    resolver.set_user_priority("senior_analyst", 3)
    resolver.set_user_priority("junior_analyst", 1)
    resolver.set_user_priority("reviewer", 2)
    
    # Set content importance
    resolver.set_content_importance("valuation", 2.0)
    resolver.set_content_importance("assumption", 1.5)
    resolver.set_content_importance("note", 0.5)
    
    # 3. Real-time Financial Calculator
    print("\n3. Setting up real-time financial calculator...")
    calculator = RealTimeFinancialCalculator()
    await calculator.start()
    
    # Set up calculation dependencies
    calculator.add_calculation_dependency("weighted_average", 
                                        ["dcf_valuation", "comparable_analysis", "precedent_transactions"])
    
    # Queue calculations
    await calculator.queue_calculation("dcf_valuation", "dcf_valuation", {
        "revenue": 150000000,
        "growth_rate": 0.08,
        "discount_rate": 0.12,
        "terminal_growth": 0.03,
        "projection_years": 5
    }, "workspace_123")
    
    await calculator.queue_calculation("comparable_analysis", "comparable_analysis", {
        "comparables": [
            {"revenue": 120000000, "enterprise_value": 600000000},
            {"revenue": 180000000, "enterprise_value": 720000000},
            {"revenue": 90000000, "enterprise_value": 450000000}
        ],
        "target_metric": "revenue",
        "target_value": 150000000
    }, "workspace_123")
    
    await calculator.queue_calculation("precedent_transactions", "precedent_transactions", {
        "transactions": [
            {"revenue": 100000000, "transaction_value": 500000000},
            {"revenue": 200000000, "transaction_value": 800000000}
        ],
        "target_metric": "revenue", 
        "target_value": 150000000
    }, "workspace_123")
    
    # Wait for calculations to complete
    await asyncio.sleep(5)
    
    # Queue weighted average calculation
    await calculator.queue_calculation("weighted_average", "weighted_average", {
        "weights": {
            "dcf_valuation": 0.5,
            "comparable_analysis": 0.3,
            "precedent_transactions": 0.2
        }
    }, "workspace_123")
    
    # Wait for final calculation
    await asyncio.sleep(2)
    
    # Show results
    print("\nCalculation Results:")
    for calc_id, result in calculator.cached_results.items():
        print(f"{calc_id}: ${result['enterprise_value']:,.0f}")
    
    await calculator.stop()
    
    print("\n=== Advanced Features Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(advanced_features_example())