from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat
from fastapi.templating import Jinja2Templates
import os


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



templates = Jinja2Templates(directory=os.path.join(BASE_DIR,"templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:8501"] or a specific route
    allow_credentials=True,
    allow_methods=["*"],  # Or specify ["GET", "POST"]
    allow_headers=["*"],  # Or specify ["Content-Type", "Authorization"]
)







app.include_router(chat.router)

@app.get('/')
async def root(request:Request):
    return templates.TemplateResponse("index.html", {"request":request})












