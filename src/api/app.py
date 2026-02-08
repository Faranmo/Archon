"""
Archon FastAPI Application

Main application factory with middleware and lifecycle management.
"""

import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.config import settings
from src.core.errors import ArchonError
from src.monitoring.logger import (
    get_logger,
    set_request_context,
    clear_request_context,
    api_logger,
)
from src.security.rate_limiter import rate_limit_middleware_helper

logger = get_logger("api.app")


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown.
    """
    # Startup
    logger.info(
        "Starting Archon API",
        metadata={
            "environment": settings.environment,
            "debug": settings.debug,
        }
    )

    yield

    # Shutdown
    logger.info("Shutting down Archon API")


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Archon API",
        description="Production-grade AI Research Assistant",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Add middleware
    _configure_middleware(app)

    # Add routes
    _configure_routes(app)

    # Add exception handlers
    _configure_exception_handlers(app)

    return app


def _configure_middleware(app: FastAPI):
    """Configure application middleware."""

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging and context
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next: Callable) -> Response:
        """Log requests and set context."""
        start_time = time.time()

        # Set request context
        request_id = request.headers.get("X-Request-ID")
        context = set_request_context(
            request_id=request_id,
            user_id=request.headers.get("X-User-ID"),
        )

        # Log request
        api_logger.info(
            f"{request.method} {request.url.path}",
            metadata={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
            }
        )

        try:
            response = await call_next(request)

            # Log response
            latency_ms = (time.time() - start_time) * 1000
            api_logger.info(
                f"Response {response.status_code}",
                metadata={
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                }
            )

            # Add headers
            response.headers["X-Request-ID"] = context["request_id"]
            response.headers["X-Response-Time"] = f"{latency_ms:.2f}ms"

            return response

        except Exception as e:
            api_logger.error(
                f"Request failed: {e}",
                metadata={"error": str(e)}
            )
            raise

        finally:
            clear_request_context()

    # Rate limiting
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
        """Apply rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/ready"):
            return await call_next(request)

        try:
            headers = await rate_limit_middleware_helper(
                ip_address=request.client.host if request.client else None,
                api_key=request.headers.get("X-API-Key"),
                endpoint=request.url.path,
            )

            response = await call_next(request)

            # Add rate limit headers
            for key, value in headers.items():
                response.headers[key] = value

            return response

        except Exception as e:
            # Rate limit exceeded
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": 60},
            )


def _configure_routes(app: FastAPI):
    """Configure application routes."""
    from src.api.routes import router

    app.include_router(router, prefix="/api/v1")

    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "1.0.0"}

    @app.get("/ready")
    async def readiness_check():
        """Readiness check endpoint."""
        # Add dependency checks here
        return {"status": "ready"}


def _configure_exception_handlers(app: FastAPI):
    """Configure exception handlers."""

    @app.exception_handler(ArchonError)
    async def archon_error_handler(request: Request, exc: ArchonError) -> JSONResponse:
        """Handle Archon errors."""
        api_logger.error(
            f"ArchonError: {exc.message}",
            metadata=exc.to_dict()
        )

        status_code = 400
        if exc.category.value == "authentication_error":
            status_code = 401
        elif exc.category.value == "authorization_error":
            status_code = 403
        elif exc.category.value == "rate_limit":
            status_code = 429
        elif exc.category.value in ("timeout", "network_error"):
            status_code = 503

        return JSONResponse(
            status_code=status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        api_logger.error(
            f"Unhandled exception: {exc}",
            metadata={"type": type(exc).__name__}
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(exc) if settings.debug else "An unexpected error occurred",
            },
        )


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()
