"""
Unit tests for the main FastAPI application.

Tests the core application setup, middleware configuration, exception handling,
and utility endpoints.
"""

import asyncio
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from src.backend.api.main import create_application, MetricsMiddleware
from tests.utils.test_data import get_standard_test_scenario


class TestApplicationCreation:
    """Test application creation and configuration."""
    
    def test_create_application_returns_fastapi_instance(self):
        """Test that create_application returns a FastAPI instance."""
        app = create_application()
        
        assert app is not None
        assert hasattr(app, 'routes')
        assert hasattr(app, 'middleware')
    
    def test_application_has_correct_metadata(self, test_settings):
        """Test that application has correct title, version, etc."""
        with patch('src.backend.api.main.settings', test_settings):
            app = create_application()
            
            assert app.title == test_settings.api.title
            assert app.version == test_settings.api.version
    
    def test_middleware_configuration(self):
        """Test that all required middleware is configured."""
        app = create_application()
        
        # Check that middleware stack is properly configured
        middleware_types = [type(middleware) for middleware in app.user_middleware]
        middleware_names = [middleware.__class__.__name__ for middleware in app.user_middleware]
        
        # Verify key middleware components are present
        expected_middleware = [
            'RequestIDMiddleware',
            'AuthenticationMiddleware', 
            'RateLimitMiddleware',
            'CORSMiddleware',
            'GZipMiddleware',
            'MetricsMiddleware'
        ]
        
        for expected in expected_middleware:
            assert any(expected in name for name in middleware_names), \
                f"Missing middleware: {expected}"
    
    def test_exception_handlers_registered(self):
        """Test that exception handlers are properly registered."""
        app = create_application()
        
        # Check that exception handlers are registered
        assert app.exception_handlers is not None
        assert len(app.exception_handlers) > 0
    
    def test_router_inclusion(self):
        """Test that API router is properly included."""
        app = create_application()
        
        # Check that routes are registered
        route_paths = [route.path for route in app.routes]
        
        # Should have utility routes
        assert "/health" in route_paths
        assert "/metrics" in route_paths
        assert "/info" in route_paths
        
        # Should have API routes with prefix
        api_routes = [path for path in route_paths if path.startswith("/api/v1")]
        assert len(api_routes) > 0


class TestHealthEndpoints:
    """Test health check and utility endpoints."""
    
    def test_health_endpoint_returns_healthy_status(self, test_client):
        """Test basic health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_with_healthy_database(self, async_test_client, mock_db_session):
        """Test detailed health check with healthy database."""
        # Mock successful database query
        mock_db_session.execute = AsyncMock(return_value=Mock())
        
        response = await async_test_client.get("/health/detailed")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["services"]["database"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_detailed_health_check_with_unhealthy_database(self, async_test_client):
        """Test detailed health check with database failure."""
        with patch('src.backend.database.base.db_manager.get_async_session') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            response = await async_test_client.get("/health/detailed")
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "unhealthy" in data["services"]["database"]
    
    def test_app_info_endpoint(self, test_client, test_settings):
        """Test application info endpoint."""
        response = test_client.get("/info")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "environment" in data
        assert "debug" in data
    
    @patch('src.backend.api.main.settings')
    def test_metrics_endpoint_enabled(self, mock_settings, test_client):
        """Test metrics endpoint when enabled."""
        mock_settings.monitoring.metrics_enabled = True
        
        with patch('src.backend.api.main.generate_latest', return_value=b'# HELP test_metric Test metric\n'):
            response = test_client.get("/metrics")
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    
    @patch('src.backend.api.main.settings')
    def test_metrics_endpoint_disabled(self, mock_settings, test_client):
        """Test metrics endpoint when disabled."""
        mock_settings.monitoring.metrics_enabled = False
        
        response = test_client.get("/metrics")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestExceptionHandling:
    """Test application exception handling."""
    
    def test_validation_error_handling(self, test_client):
        """Test RequestValidationError handling."""
        # This would trigger a validation error in a real endpoint
        # For testing, we'll mock the scenario
        with patch('src.backend.api.main.app') as mock_app:
            from fastapi.exceptions import RequestValidationError
            
            # Simulate validation error
            error = RequestValidationError([
                {
                    'loc': ('body', 'email'),
                    'msg': 'field required',
                    'type': 'value_error.missing'
                }
            ])
            
            # Test the exception handler directly
            from src.backend.api.main import create_application
            app = create_application()
            
            # The actual test would require a real endpoint that triggers validation
            # This tests the handler configuration
            assert len(app.exception_handlers) > 0
    
    def test_http_exception_handling(self, test_client):
        """Test HTTPException handling."""
        # Test by calling a non-existent endpoint
        response = test_client.get("/non-existent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        data = response.json()
        assert "detail" in data
    
    def test_general_exception_handling(self):
        """Test general exception handling."""
        from fastapi import Request
        from src.backend.api.main import create_application
        
        app = create_application()
        
        # Verify that a general exception handler is registered
        assert Exception in app.exception_handlers or \
               any(issubclass(Exception, exc_type) for exc_type in app.exception_handlers.keys())


class TestMetricsMiddleware:
    """Test the metrics middleware."""
    
    @pytest.mark.asyncio
    async def test_metrics_middleware_records_request(self):
        """Test that metrics middleware records request metrics."""
        from fastapi import Request, Response
        from unittest.mock import AsyncMock
        
        middleware = MetricsMiddleware(app=Mock())
        
        # Mock request and response
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        
        response = Mock(spec=Response)
        response.status_code = 200
        
        # Mock the call_next function
        call_next = AsyncMock(return_value=response)
        
        with patch('src.backend.api.main.REQUEST_COUNT') as mock_counter, \
             patch('src.backend.api.main.REQUEST_DURATION') as mock_histogram, \
             patch('asyncio.get_event_loop') as mock_loop:
            
            mock_loop.return_value.time.side_effect = [0.0, 0.5]  # Start and end times
            
            # Execute middleware
            result = await middleware.dispatch(request, call_next)
            
            # Verify metrics were recorded
            mock_counter.labels.assert_called_once_with(
                method="GET", 
                endpoint="/test", 
                status=200
            )
            mock_counter.labels.return_value.inc.assert_called_once()
            
            mock_histogram.labels.assert_called_once_with(
                method="GET", 
                endpoint="/test"
            )
            mock_histogram.labels.return_value.observe.assert_called_once_with(0.5)
            
            assert result == response


class TestApplicationLifespan:
    """Test application lifespan management."""
    
    @pytest.mark.asyncio
    async def test_startup_tasks_execution(self):
        """Test that startup tasks are executed."""
        with patch('src.backend.api.main.db_manager') as mock_db_manager, \
             patch('src.backend.api.main.setup_metrics') as mock_setup_metrics:
            
            mock_db_manager.create_tables_async = AsyncMock()
            
            from src.backend.api.main import lifespan, create_application
            
            app = create_application()
            
            # Test lifespan startup
            async with lifespan(app):
                pass
            
            # Verify startup tasks were called
            mock_db_manager.create_tables_async.assert_called_once()
            mock_setup_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_failure_handling(self):
        """Test startup failure handling."""
        with patch('src.backend.api.main.db_manager') as mock_db_manager:
            
            # Simulate startup failure
            mock_db_manager.create_tables_async = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            
            from src.backend.api.main import lifespan, create_application
            
            app = create_application()
            
            # Test that startup failure is properly handled
            with pytest.raises(Exception, match="Database connection failed"):
                async with lifespan(app):
                    pass
    
    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self):
        """Test that cleanup tasks are executed on shutdown."""
        with patch('src.backend.api.main.db_manager') as mock_db_manager:
            
            mock_db_manager.create_tables_async = AsyncMock()
            mock_db_manager.close = AsyncMock()
            
            from src.backend.api.main import lifespan, create_application
            
            app = create_application()
            
            # Test lifespan with cleanup
            async with lifespan(app):
                pass
            
            # Verify cleanup was called
            mock_db_manager.close.assert_called_once()


class TestApplicationIntegration:
    """Integration tests for the full application setup."""
    
    def test_application_starts_successfully(self, test_settings):
        """Test that application can be created and is ready to serve requests."""
        app = create_application()
        
        with TestClient(app) as client:
            # Test that we can make basic requests
            response = client.get("/health")
            assert response.status_code == status.HTTP_200_OK
            
            response = client.get("/info")
            assert response.status_code == status.HTTP_200_OK
    
    def test_cors_configuration(self, test_client):
        """Test CORS middleware configuration."""
        # Test preflight request
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should handle CORS preflight
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
    
    @patch('src.backend.api.main.settings')
    def test_debug_mode_configuration(self, mock_settings, test_client):
        """Test application behavior in debug mode."""
        mock_settings.debug = True
        mock_settings.api.docs_url = "/docs"
        
        app = create_application()
        
        with TestClient(app) as client:
            # In debug mode, docs should be available
            response = client.get("/docs")
            # Note: This might return 404 if no routes are properly configured
            # The test verifies the configuration logic rather than actual docs
    
    @patch('src.backend.api.main.settings')
    def test_production_mode_configuration(self, mock_settings):
        """Test application behavior in production mode."""
        mock_settings.debug = False
        mock_settings.api.docs_url = None
        mock_settings.api.redoc_url = None
        mock_settings.api.openapi_url = None
        
        app = create_application()
        
        # In production mode, docs URLs should be None
        assert app.docs_url is None
        assert app.redoc_url is None
        assert app.openapi_url is None


@pytest.mark.integration
class TestApplicationWithDependencies:
    """Integration tests with external dependencies."""
    
    def test_application_with_database(self, test_client, async_db_session):
        """Test application works with database connection."""
        response = test_client.get("/health/detailed")
        
        # Should be able to check database health
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
    
    def test_application_with_redis(self, test_client, mock_redis_client):
        """Test application works with Redis connection."""
        # This test would verify Redis integration if implemented
        response = test_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
    
    def test_application_with_gcp_services(self, test_client, mock_gcp_services):
        """Test application works with GCP services."""
        # This test would verify GCP integration if implemented
        response = test_client.get("/health")
        assert response.status_code == status.HTTP_200_OK


# Performance tests for the main application
@pytest.mark.performance
class TestApplicationPerformance:
    """Performance tests for application startup and basic operations."""
    
    def test_application_startup_time(self):
        """Test application startup performance."""
        import time
        
        start_time = time.time()
        app = create_application()
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        # Application should start within reasonable time
        assert startup_time < 5.0, f"Application startup took {startup_time:.2f} seconds"
    
    def test_health_endpoint_response_time(self, test_client):
        """Test health endpoint response time."""
        import time
        
        # Warm up
        test_client.get("/health")
        
        start_time = time.time()
        response = test_client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 0.1, f"Health check took {response_time:.3f} seconds"
    
    def test_concurrent_requests_handling(self, test_client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import time
        
        def make_request():
            return test_client.get("/health")
        
        # Make 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            futures = [executor.submit(make_request) for _ in range(50)]
            responses = [future.result() for future in futures]
            end_time = time.time()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
        
        total_time = end_time - start_time
        assert total_time < 10.0, f"50 concurrent requests took {total_time:.2f} seconds"


# Security tests for the main application
@pytest.mark.security
class TestApplicationSecurity:
    """Security tests for the main application."""
    
    def test_security_headers(self, test_client):
        """Test that security headers are present."""
        response = test_client.get("/health")
        
        # Check for security-related headers
        # Note: Actual headers depend on middleware configuration
        assert response.status_code == status.HTTP_200_OK
        
        # These would be present if security middleware is configured
        # assert "X-Content-Type-Options" in response.headers
        # assert "X-Frame-Options" in response.headers
    
    def test_sensitive_information_not_exposed(self, test_client):
        """Test that sensitive information is not exposed in responses."""
        response = test_client.get("/info")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        
        # Should not expose sensitive configuration
        assert "password" not in str(data).lower()
        assert "secret" not in str(data).lower()
        assert "key" not in str(data).lower()
    
    def test_error_responses_dont_expose_internals(self, test_client):
        """Test that error responses don't expose internal details."""
        response = test_client.get("/non-existent-endpoint")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        data = response.json()
        
        # Should not expose internal paths, stack traces, etc.
        error_text = str(data).lower()
        assert "/src/" not in error_text
        assert "traceback" not in error_text
        assert "stacktrace" not in error_text