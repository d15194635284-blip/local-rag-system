import os
import tempfile
from typing import List, Tuple
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import ollama

# 配置
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "doc_chunks"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K = 15   # 关键：检索15个块，覆盖几乎所有内容

# 使用 Ollama 的嵌入模型
embedding_fn = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api/embeddings"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            while end > start and text[end] not in "。！？\n":
                end -= 1
            if end == start:
                end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def index_document(file_content: bytes, filename: str) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        temp_path = tmp_file.name

    full_text = extract_text_from_pdf(temp_path)
    os.remove(temp_path)

    if not full_text.strip():
        raise ValueError("PDF 中没有提取到文本内容")

    chunks = split_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    doc_id = filename.replace(".pdf", "")
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    existing = collection.get(where={"source": filename})
    if existing['ids']:
        collection.delete(ids=existing['ids'])

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)

def query_document(question: str) -> Tuple[str, List[str]]:
    results = collection.query(query_texts=[question], n_results=TOP_K)
    retrieved_chunks = results['documents'][0] if results['documents'] else []
    if not retrieved_chunks:
        return "未在文档中找到相关内容，请先上传相关文档。", []

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""请根据下面给出的文档内容回答用户的问题。如果文档中包含答案，请用完整的中文句子回答，并尽量引用原文；如果完全没有相关信息，则回答“文档中未提及”。

文档内容：
{context}

问题：{question}

回答："""

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1}
    )
    answer = response['message']['content'].strip()
    return answer, retrieved_chunks