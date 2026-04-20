from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("FACE_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("FACE_SERVICE_PORT", "8003"))
    uvicorn.run("face_service.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

