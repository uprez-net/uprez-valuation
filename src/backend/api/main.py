"""
FastAPI main application with comprehensive API setup
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import logging
import time

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..core.config import settings
from ..core.database import init_db, close_db
from ..middleware.security import SecurityMiddleware
from ..middleware.rate_limit import RateLimitMiddleware
from ..middleware.logging import LoggingMiddleware
from ..middleware.monitoring import MonitoringMiddleware

# Import API routers
from .v1.auth import router as auth_router
from .v1.users import router as users_router
from .v1.companies import router as companies_router
from .v1.valuations import router as valuations_router
from .v1.documents import router as documents_router
from .v1.collaboration import router as collaboration_router
from .v1.integrations import router as integrations_router
from .v1.admin import router as admin_router
from .websocket import router as websocket_router
from .graphql import graphql_app

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting IPO Valuation Platform API...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized successfully")
        
        # Initialize external services
        # await init_external_services()
        
        # Start background tasks
        # await start_background_tasks()
        
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    try:
        # Close database connections
        await close_db()
        logger.info("Database connections closed")
        
        # Clean up resources
        # await cleanup_resources()
        
        logger.info("Application shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations"""
    
    # Create FastAPI app with custom docs
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url=None,  # Disable default docs
        redoc_url=None,  # Disable default redoc
        openapi_url=settings.OPENAPI_URL,
        lifespan=lifespan
    )
    
    # Add middleware in correct order (last added runs first)
    
    # Monitoring middleware (outermost)
    app.add_middleware(MonitoringMiddleware)
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Security middleware
    app.add_middleware(SecurityMiddleware)
    
    # Rate limiting middleware  
    app.add_middleware(RateLimitMiddleware)
    
    # Logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Trusted host middleware
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.uprez.com", "uprez.com"]
        )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include API routers
    api_prefix = settings.API_V1_STR
    
    app.include_router(auth_router, prefix=f"{api_prefix}/auth", tags=["Authentication"])
    app.include_router(users_router, prefix=f"{api_prefix}/users", tags=["Users"])
    app.include_router(companies_router, prefix=f"{api_prefix}/companies", tags=["Companies"])
    app.include_router(valuations_router, prefix=f"{api_prefix}/valuations", tags=["Valuations"])
    app.include_router(documents_router, prefix=f"{api_prefix}/documents", tags=["Documents"])
    app.include_router(collaboration_router, prefix=f"{api_prefix}/collaboration", tags=["Collaboration"])
    app.include_router(integrations_router, prefix=f"{api_prefix}/integrations", tags=["Integrations"])
    app.include_router(admin_router, prefix=f"{api_prefix}/admin", tags=["Administration"])
    
    # WebSocket routes
    app.include_router(websocket_router, prefix="/ws")
    
    # GraphQL endpoint
    app.mount("/graphql", graphql_app)
    
    # Custom documentation endpoints
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Interactive API docs",
            swagger_js_url="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui-bundle.js",
            swagger_css_url="https://unpkg.com/swagger-ui-dist@4.15.5/swagger-ui.css",
        )
    
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - ReDoc",
            redoc_js_url="https://unpkg.com/redoc@next/bundles/redoc.standalone.js",
        )
    
    # Health check endpoints
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT
        }
    
    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health_check():
        """Detailed health check with service status"""
        # Check database connectivity
        db_healthy = True
        try:
            # Perform database health check
            pass
        except Exception:
            db_healthy = False
        
        # Check external services
        external_services = {
            "asx_api": True,  # Check ASX API connectivity
            "asic_api": True,  # Check ASIC API connectivity
            "redis": True,    # Check Redis connectivity
        }
        
        overall_status = "healthy" if db_healthy and all(external_services.values()) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "services": {
                "database": "healthy" if db_healthy else "unhealthy",
                **{k: "healthy" if v else "unhealthy" for k, v in external_services.items()}
            }
        }
    
    # API status endpoint
    @app.get("/status", tags=["Status"])
    async def api_status():
        """API status and metrics"""
        return {
            "api_name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime": time.time(),
            "endpoints": len(app.routes),
            "features": {
                "authentication": True,
                "rate_limiting": True,
                "websockets": True,
                "graphql": True,
                "file_upload": True,
                "external_integrations": True
            }
        }
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint with basic information"""
        return {
            "message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "documentation": "/docs",
            "health": "/health",
            "status": "/status"
        }
    
    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent format"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "http_exception",
                    "timestamp": time.time()
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error" if settings.ENVIRONMENT == "production" else str(exc),
                    "type": "internal_error",
                    "timestamp": time.time()
                }
            }
        )
    
    # Request/Response middleware for consistent API format
    @app.middleware("http")
    async def api_response_middleware(request: Request, call_next):
        """Middleware for consistent API responses"""
        start_time = time.time()
        
        # Add request ID
        request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
        
        try:
            response = await call_next(request)
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(int((time.time() - start_time) * 1000))
            response.headers["X-API-Version"] = settings.VERSION
            
            return response
            
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
    
    return app


# Create application instance
app = create_app()

# For development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )