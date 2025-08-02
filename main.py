import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.websocket("/ws")
async def chat_websocket(ws: WebSocket):
    await ws.accept()
    chat = model.start_chat(history=[])
    try:
        while True:
            user_msg = await ws.receive_text()
            if user_msg.lower() in ["exit", "quit"]:
                await ws.close()
                break

            # Send message to Gemini
            response = chat.send_message(user_msg, stream=True)
            text = ""
            for chunk in response:
                text_chunk = chunk.text
                text += text_chunk
                await ws.send_text(text_chunk)
    except WebSocketDisconnect:
        pass
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
