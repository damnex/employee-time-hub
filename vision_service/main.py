from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("VISION_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("VISION_SERVICE_PORT", "8002"))
    uvicorn.run("vision_service.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
