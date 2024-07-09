import os
import torch

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM

Llama_tokenizer_model_path = "/home/patchy/Model/models/Llama-2-7b-hf"

quant_model_path = "/home/patchy/Model/models/mergered_chn_llama/"


class EmbedChunks:
    def __init__(self, model_name):
        # Embedding model
        self.get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": "cuda", "fast_tokenizer": "1"},
            encode_kwargs={"device": "cuda", "batch_size": 100},
        )

    def __call__(self, batch):
        embeddings = self.GenerateEmbeddings(self.embedding_model, self.tokenizer , batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}
    
    
    def GenerateEmbeddings(self, embedding_model, tokenizer, text):
        # token
        # input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].to(self.device)
        
        
        with torch.no_grad():
            embeddings = embedding_model(**encoded_input)
        return embeddings
    
    
    
    def get_embedding_model(self, embedding_model_name, model_kwargs, encode_kwargs) -> None:
        # if embedding_model_name == "text-embedding-ada-002":
        #     embedding_model = OpenAIEmbeddings(
        #         model=embedding_model_name,
        #         openai_api_base=os.environ["OPENAI_API_BASE"],
        #         openai_api_key=os.environ["OPENAI_API_KEY"],
        #     )
        # else:
        #     embedding_model = HuggingFaceEmbeddings(
        #         model_name=embedding_model_name,
        #         model_kwargs=model_kwargs,
        #         encode_kwargs=encode_kwargs,
        #     )
            
        device = model_kwargs.get("device", "cpu") if torch.cuda.is_available() else "cpu"
        self.device = device
        fast_tokenizer = model_kwargs.get("fast_tokenizer", "0")
        tokenizer = AutoTokenizer.from_pretrained(quant_model_path, use_fast=True if fast_tokenizer else False)
        self.tokenizer = tokenizer
            
        model = AutoModelForCausalLM.from_pretrained(embedding_model_name, torch_dtype=torch.float16)
        
        model.to(device)
        
        model.eval()
        
        # print model to get embedding model name
        embedding_model = model.model.embed_tokens
        
        self.embedding_model = embedding_model
    
    def PrintEmbed(self):
        print(self.embedding_model)
        print("----------------------------------------")
        print(self.tokenizer)
        print("----------------------------------------")
        print(self.device)
