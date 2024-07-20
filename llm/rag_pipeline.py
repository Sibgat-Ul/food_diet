import os
import json
import requests
import weaviate
import regex as re
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey

from llm.model import LLMPipeline
from llm.PhiPrompt import PhiPrompt

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda
)
from search.SearchWeb import TavilySearch, DDGSearch, MultiSearch


def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

def load_store():
    pass


class Chain:
    def __init__(self, model_id: str, api_key: dict, device: str = 'cuda'):
        self.model_id = model_id
        self.phi_prompt_builder = PhiPrompt()
        self.pipe = LLMPipeline(model_id=model_id, device=device)
        self.llm = HuggingFacePipeline(
            pipeline=self.pipe.get_pipeline()
        )

        self.search = MultiSearch(
            [
                TavilySearch(api_key=api_key['tavily']),
                DDGSearch()
            ]
        )

    def _prompt_template(self, system_prompt_text: str, user_prompt_text: str, input_vars: []) -> PromptTemplate:
        system_prompt = self.phi_prompt_builder.system_prompt(system_prompt_text)
        user_prompt = self.phi_prompt_builder.user_prompt(user_prompt_text)
        assistant_prompt = self.phi_prompt_builder.assistant_prompt()

        prompt = system_prompt + ' ' + user_prompt + ' ' + assistant_prompt

        return PromptTemplate(template=prompt, input_variables=input_vars)

    def _search_prompt(self, prompt):
        system_prompt_text = """You are a helpful assistant. You will assist me by generating queries and returning \
the queries in a list format to search the web about recipes for given context."""

        user_prompt_text = f"""
        Generate 3 queries to find food recipe for context: '{prompt}'
        """

        return self._prompt_template(system_prompt_text, user_prompt_text, [])

    def _generate_search(self, prompt: str) -> list:
        search_prompt = self._search_prompt(prompt)
        web_queries = ((search_prompt
                       | self.llm
                       | StrOutputParser()
                       | (lambda x: (x.split("<|assistant|>")[-1]).strip()))
                       | (lambda x: (re.findall(r'[1-9]\.\s?"(.*)"\n?', x))))

        return web_queries.invoke({})

    def _query_web(self, queries: list):
        return [self.search.search(q) for q in queries]

    def search_chain(self, prompt: str):
        web_queries = self._generate_search(prompt)
        web_queries.append(prompt)

        search_results = self._query_web(web_queries)

        return search_results

    def _llm_prompt(self) -> PromptTemplate:
        system_prompt_text = """You are an expert dietitian also a great cook. Your task is to produce diet plan \
according to the patient's requirement and give your recipe to cook your prescribed diet. You will be provided \
with the data of nutritional values of the food ingredients, use those values to generate the diet plans and food \
recipe incorporating the nutritional values for the diet. Nutritional values are {nutritional_values}. Your generated \
output should have calorie contents of each recipes."""

        user_prompt_text = """generate diet along with recipe for the requirement: {prompt}"""

        return self._prompt_template(system_prompt_text, user_prompt_text, ['prompt', 'nutritional_values'])

    def prompt_chain(self, prompt: str, nutritional_values):
        chain = self._llm_prompt() | self.llm | StrOutputParser()

        res = chain.invoke({'prompt': prompt, 'nutritional_values': nutritional_values})
        return res
