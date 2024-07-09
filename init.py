import os
import ray
import sys; sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
from dotenv import load_dotenv; load_dotenv()
from rag.config import EFS_DIR
from pathlib import Path
from rag.config import ROOT_DIR
from rag.config import EMBEDDING_DIMENSIONS, MAX_CONTEXT_LENGTHS, PRICING
from rag.data import extract_sections

from functools import partial
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Credentials
ray.init(runtime_env={
    "env_vars": {
        # "OPENAI_API_BASE": os.environ["OPENAI_API_BASE"],
        # "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"], 
        # "ANYSCALE_API_BASE": os.environ["ANYSCALE_API_BASE"],
        # "ANYSCALE_API_KEY": os.environ["ANYSCALE_API_KEY"],
        "DB_CONNECTION_STRING": os.environ["DB_CONNECTION_STRING"],
    },
    "working_dir": str(ROOT_DIR)
})


ray.cluster_resources()


# Ray dataset
DOCS_DIR = Path(EFS_DIR, "xiaolincoding.com/")
ds = ray.data.from_items([{"path": path} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()])
print(f"{ds.count()} documents")
print(f"{ds.show(2)}")


# Extract sections
sections_ds = ds.flat_map(extract_sections)
sections_ds.count()

# section_lengths = []
# for section in sections_ds.take_all():
#     section_lengths.append(len(section["text"]))

# Text splitter
chunk_size = 256
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", "", "；", "。"],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len)


# Chunk a sample section
sample_section = sections_ds.take(1)[0]
chunks = text_splitter.create_documents(
    texts=[sample_section["text"]], 
    metadatas=[{"source": sample_section["source"]}])
print (chunks[0])