from fastapi import FastAPI

from .routes import router

app = FastAPI(title="ReviewScope", description="Multilingual review sentiment classfier")
app.include_router(router)
