### About:

This is a RAG application to generate diet plan. It searches the web to generate diet plans according to the prompt and needs of the user. 

<br>
It utilizes,
<br>

1. Microsoft Phi-3 model to generate the replies (quantized)
2. Huggingface pipeline to infer
3. Chroma-db vector store as retriever
4. Mini LM embedding model 
5. Tavily search api to search the web

### Install:

```bash
pip install -r 'requirements.txt'
```

### To run:

1. Create a .env file and add
   1. TAVILY_API_KEY=your_tavily_api_key_(check their website, its free)

2. Firstly start the fastapi server:
```bash
uvicorn main:app --reload --reload-exclude ./front_end/*
```

3. Start the streamlit server:
```bash
streamlit run front_end/streamlit_ui.py
```

### Future work:

1. Cloud based vector store
2. SQL based Chat history retrieving 
3. Modify the whole structure so that any model can be used.