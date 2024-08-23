import torch
import sentencepiece

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from langchain_huggingface import HuggingFacePipeline


class LLMPipeline:
    def __init__(self, model_id: str, device: str = 'cuda'):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=self.bnb_config,
        )

        self.pipe = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
        )

    def get_info(self):
        print(f'model_info: {self.model}')

    def get_pipeline(self) -> HuggingFacePipeline:
        return HuggingFacePipeline(pipeline=self.pipe)
