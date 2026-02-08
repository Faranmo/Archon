"""
Archon - Production-Grade AI Research Assistant

Main entry point for running the application.
"""

import uvicorn
from src.core.config import settings


def main():
    """Run the Archon API server."""
    uvicorn.run(
        "src.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
