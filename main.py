import os

from llm.rag_pipeline import Chain
import dotenv

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

    res = rag_chain.search_chain('look for biriyani')
    print(res)

if __name__ == '__main__':
    main()
    print('Hello World!')
