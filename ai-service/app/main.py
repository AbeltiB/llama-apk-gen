"""
Main FastAPI application - Updated with enterprise features.

New features:
1. REST API endpoint for frontend (/api/v1/generate)
2. Enterprise structured logging with correlation tracking
3. Multi-tier health checks (liveness, readiness, full)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import time

from app.config import settings
from app.core.cache import cache_manager
from app.core.database import db_manager
from app.api.v1 import stats
from app.api.v1 import results
from app.api.v1 import components

# Import new structured logging
from app.utils.logging import get_logger, log_context

# Import routers
from app.api.v1 import health, generate

logger = get_logger(__name__)


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with structured logging"""
    
    correlation_id = str(uuid.uuid4())
    
    with log_context(correlation_id=correlation_id, operation="startup"):
        logger.info(
            "app.startup.started",
            extra={
                "service": settings.app_name,
                "version": settings.app_version,
                "debug": settings.debug
            }
        )
        
        logger.info("app.startup.celery.mode", extra={"transport": "redis"})

        # Redis
        try:
            await cache_manager.connect()
            logger.info("app.startup.redis.connected")
        except Exception as e:
            logger.error("app.startup.redis.failed", exc_info=e)
            logger.warning("app.startup.redis.degraded_mode", message="Continuing without cache")
        
        # PostgreSQL
        try:
            await db_manager.connect()
            logger.info("app.startup.postgresql.connected")
        except Exception as e:
            logger.critical("app.startup.postgresql.failed", exc_info=e)
            raise
        
        logger.info(
            "app.startup.completed",
            extra={"status": "ready"}
        )
        
        yield
        
        # Shutdown
        with log_context(correlation_id=str(uuid.uuid4()), operation="shutdown"):
            logger.info("app.shutdown.started")
            
            await cache_manager.disconnect()
            logger.info("app.shutdown.redis.disconnected")
            
            await db_manager.disconnect()
            logger.info("app.shutdown.postgresql.disconnected")
            
            logger.info("app.shutdown.completed")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered mobile app generation service with enterprise features",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE LOGGING MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all HTTP requests with correlation tracking"""
    
    # Generate correlation ID
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    
    start_time = time.time()
    
    with log_context(
        correlation_id=correlation_id,
        endpoint=request.url.path,
        method=request.method
    ):
        logger.info(
            "http.request.received",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        try:
            response = await call_next(request)
            
            duration_ms = (time.time() - start_time) * 1000
            
            logger.performance(
                "http.request.completed",
                duration_ms=duration_ms,
                extra={
                    "status_code": response.status_code,
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            logger.error(
                "http.request.failed",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "duration_ms": duration_ms
                },
                exc_info=e
            )
            raise


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with structured logging"""
    
    logger.error(
        "app.exception.unhandled",
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        },
        exc_info=exc
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "correlation_id": request.headers.get("X-Correlation-ID", "unknown")
        }
    )


# ============================================================================
# ROUTERS
# ============================================================================

# Health checks (multi-tier)
app.include_router(
    health.router,
    tags=["Health"]
)

# Generation endpoint (frontend API)
app.include_router(
    generate.router,
    prefix="/api/v1",
    tags=["Generation"]
)

# Results endpoint
app.include_router(
    results.router,
    prefix="/api/v1",
    tags=["Results"]
)

# Statistics endpoints
app.include_router(
    stats.router,
    prefix="/api/v1",
    tags=["Statistics"]
)

# Component catalog endpoints
app.include_router(
    components.router,
    prefix="/api/v1",
    tags=["Components"]
)

# Test routes (development only)
if settings.debug:
    from app.api.v1 import test_routes
    app.include_router(
        test_routes.router,
        prefix="/api/v1",
        tags=["Testing"]
    )

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": {
            "liveness": "/health/live",
            "readiness": "/health/ready",
            "full": "/health/full"
        },
        "api": {
            "generate": "POST /api/v1/generate"
        }
    }


# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(
        "app.dev_server.starting",
        extra={
            "host": "0.0.0.0",
            "port": 8000,
            "reload": settings.debug
        }
    )
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )