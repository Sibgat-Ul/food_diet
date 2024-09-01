### About:

This is a RAG application to generate diet plan. It searches the web to generate diet plans according to the prompt and needs of the user. 

<br>
It utilizes,
<br>

1. Huggingface pipeline to infer
2. Chroma-db vector store as retriever
3. Minilm embedding model 
4. Tavily search api to search the web

### Install:

```bash
pip install -r 'requirements.txt'
```

### To run:

1. Firstly start the fastapi server:
```bash
uvicorn main:app --reload --reload-exclude ./front_end/*
```

2. Start the streamlit server:
```bash
streamlit run front_end/streamlit_ui.py
```

### Future work:

1. Cloud based vector store
2. SQL based Chat history retrieving 