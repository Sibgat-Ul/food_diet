import os

import dotenv
import fastapi

from back_end.server.api_model import PromptInput
from back_end.llm.rag_pipeline import Chain

dotenv.load_dotenv()

model_id = 'microsoft/Phi-3-mini-128k-instruct'
api_key = {
    'tavily': os.getenv('TAVILY_API_KEY'),
    'hf_k': os.getenv('HF_API_KEY'),
    'wv_k': os.getenv('WEAVIATE_API_KEY'),
    'wv_url': os.getenv('WEAVIATE_URL')
}

rag_chain = Chain(model_id=model_id, api_key=api_key)

app = fastapi.FastAPI()

@app.get('/')
def home():
    return "hello world"

@app.post('/prompt')
def prompt_llm(prompt: PromptInput):
    return rag_chain.prompt_chain(prompt.input)