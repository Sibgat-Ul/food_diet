import os

import dotenv
import fastapi
from fastapi import FastAPI

from llm.rag_pipeline import Chain, embed_test

dotenv.load_dotenv()


def main():
    model_id = 'microsoft/Phi-3-mini-128k-instruct'
    api_key = {
        'tavily': os.getenv('TAVILY_API_KEY'),
        'hf_k': os.getenv('HF_API_KEY'),
        'wv_k': os.getenv('WEAVIATE_API_KEY'),
        'wv_url': os.getenv('WEAVIATE_URL')
    }

    rag_chain = Chain(model_id=model_id, api_key=api_key)
    res = rag_chain.prompt_chain('diet plan to workout')
    print(res)



if __name__ == '__main__':
    main()
    print('Hello World!')
