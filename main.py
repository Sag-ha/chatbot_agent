# =============================================================================
# main.py — Project entry point
#
# Run the server locally:
#   python main.py
#
# Run with auto-reload during development:
#   uvicorn src.api:app --reload --port 8080
#
# This file must sit at the PROJECT ROOT so that Python can find both
# the `config` module and the `src` package correctly.
# =============================================================================

import uvicorn
from config import settings

if __name__ == "__main__":
    print(f"Starting Insurance Agent on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"API docs available at: http://localhost:{settings.API_PORT}/docs")

    uvicorn.run(
        "src.api:app",           # Points to the `app` object inside src/api.py
        host=settings.API_HOST,  # "0.0.0.0" — listens on all network interfaces
        port=settings.API_PORT,  # 8080 by default (Cloud Run standard)
        log_level=settings.LOG_LEVEL,
        reload=False,            # Set to True during development for auto-reload
    )
