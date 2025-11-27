from fastapi import FastAPI

from sorawm.server.lifespan import lifespan
from sorawm.server.router import router as backend_router
from sorawm.server.sora_router import router as sora_router
from sorawm.server.front_router import router as front_router



def init_app():
    app = FastAPI(lifespan=lifespan)
    app.include_router(backend_router, prefix="/api/v1")
    app.include_router(sora_router, prefix="/api/v1/sora")
    app.include_router(front_router)
    return app
