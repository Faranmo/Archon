"""
Archon API Module

FastAPI REST API with WebSocket support.
"""

from src.api.app import create_app, app
from src.api.routes import router

__all__ = [
    "create_app",
    "app",
    "router",
]
