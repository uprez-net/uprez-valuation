"""
Celery Application Configuration
Async task queue for background processing
"""
import os
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
import structlog

from ..config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Create Celery application
celery_app = Celery(
    "uprez_valuation",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=[
        'src.backend.messaging.tasks.valuation_tasks',
        'src.backend.messaging.tasks.document_tasks',
        'src.backend.messaging.tasks.ml_tasks',
        'src.backend.messaging.tasks.notification_tasks'
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    accept_content=settings.celery.accept_content,
    result_serializer=settings.celery.result_serializer,
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    
    # Task routing
    task_routes=settings.celery.task_routes,
    
    # Task execution
    task_always_eager=False,
    task_eager_propagates=True,
    task_ignore_result=False,
    task_store_eager_result=True,
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=True,
    
    # Retry configuration
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Result backend configuration
    result_expires=3600,  # 1 hour
    result_backend_always_retry=True,
    result_backend_retry_on_timeout=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security
    task_always_eager=settings.environment == "test",
)

# Queue definitions
celery_app.conf.task_routes = {
    # Valuation tasks
    'valuation.dcf_calculation': {'queue': 'valuation'},
    'valuation.cca_analysis': {'queue': 'valuation'},
    'valuation.risk_assessment': {'queue': 'valuation'},
    
    # Document processing tasks
    'document.ocr_processing': {'queue': 'document'},
    'document.sentiment_analysis': {'queue': 'document'},
    'document.entity_extraction': {'queue': 'document'},
    
    # ML tasks
    'ml.model_training': {'queue': 'ml'},
    'ml.model_inference': {'queue': 'ml'},
    'ml.data_preprocessing': {'queue': 'ml'},
    
    # Notification tasks
    'notification.send_email': {'queue': 'notifications'},
    'notification.send_webhook': {'queue': 'notifications'},
    
    # Default queue
    '*': {'queue': 'default'},
}


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready event"""
    logger.info("Celery worker ready", worker=sender.hostname)


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown event"""
    logger.info("Celery worker shutting down", worker=sender.hostname)


class CeleryTaskManager:
    """Task manager for common Celery operations"""
    
    @staticmethod
    def get_task_status(task_id: str) -> dict:
        """Get task status and result"""
        result = celery_app.AsyncResult(task_id)
        
        return {
            'task_id': task_id,
            'status': result.status,
            'result': result.result if result.ready() else None,
            'traceback': result.traceback,
            'successful': result.successful(),
            'failed': result.failed(),
            'ready': result.ready(),
        }
    
    @staticmethod
    def revoke_task(task_id: str, terminate: bool = False) -> bool:
        """Revoke a task"""
        try:
            celery_app.control.revoke(task_id, terminate=terminate)
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task {task_id}: {str(e)}")
            return False
    
    @staticmethod
    def get_active_tasks() -> list:
        """Get list of active tasks"""
        try:
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if active_tasks:
                all_tasks = []
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        task['worker'] = worker
                        all_tasks.append(task)
                return all_tasks
            
            return []
        except Exception as e:
            logger.error(f"Failed to get active tasks: {str(e)}")
            return []
    
    @staticmethod
    def get_worker_stats() -> dict:
        """Get worker statistics"""
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            return stats or {}
        except Exception as e:
            logger.error(f"Failed to get worker stats: {str(e)}")
            return {}
    
    @staticmethod
    def purge_queue(queue_name: str) -> int:
        """Purge all tasks from a queue"""
        try:
            return celery_app.control.purge()
        except Exception as e:
            logger.error(f"Failed to purge queue {queue_name}: {str(e)}")
            return 0


# Global task manager instance
task_manager = CeleryTaskManager()


if __name__ == '__main__':
    # Start Celery worker
    celery_app.start()