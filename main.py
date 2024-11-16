import os

import dotenv
import fastapi
from fastapi import WebSocket

from back_end.server.api_model import PromptInput
from back_end.llm.rag_pipeline import Chain

import uvicorn

dotenv.load_dotenv()

model_id = 'microsoft/Phi-3.5-mini-instruct'
api_key = {
    'tavily': os.getenv('TAVILY_API_KEY'),
    'hf_k': os.getenv('HF_API_KEY'),
    'wv_k': os.getenv('WEAVIATE_API_KEY'),
    'wv_url': os.getenv('WEAVIATE_URL')
}

rag_chain = Chain(model_id=model_id, api_key=api_key)

app = fastapi.FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def home():
    return "hello world"

@app.post('/prompt')
def prompt_llm(prompt: PromptInput):
    return rag_chain.prompt_chain(prompt.input)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive the input from the WebSocket client
            data = await websocket.receive_text()
            # /print(data)
            # await websocket.send_text(f"recieved: {data}")

            # Generate a response using the LLM
            response = rag_chain.prompt_chain({"input": data})
            
            # # Send the response back to the client
            await websocket.send_text(response)
    except Exception as e:
        # Handle WebSocket closure or errors gracefully
        await websocket.close()
# if __name__ == "main":
#     uvicorn.run("main:app", host="192.168.0.186", port=8080, log_level="info")