from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .tracker import VisionService, build_config_from_env


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

service = VisionService(build_config_from_env())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(title="Vision Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/tracks")
def get_tracks() -> dict[str, object]:
    return service.get_tracks()


@app.get("/status")
def get_status() -> dict[str, object]:
    return service.get_status()
