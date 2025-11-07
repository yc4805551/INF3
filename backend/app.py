# app.py
#
# 知识库后端 + AI 代理服务器 (Flask + Milvus + Ollama + Cloud APIs)
#
# 描述:
# 此版本合并了两个功能：
# 1. 知识库后端 (Milvus + Ollama 嵌入)
# 2. AI 代理 (代理对 Gemini, OpenAI, Deepseek, Ali 的调用)

import os
import click
import requests
import uuid
import time
import json # 新增
import re # 新增
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask import Flask, request, jsonify, Response, stream_with_context # 新增 Response, stream_with_context
from flask_cors import CORS
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from dotenv import load_dotenv

# --- 加载环境变量 ---
load_dotenv()

# --- 配置管理 ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_EMBED_API_URL = f"{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
KNOWLEDGE_BASE_DIR_NOMIC = os.getenv("KNOWLEDGE_BASE_DIR_NOMIC", "./knowledge_base_nomic")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", 8))

# --- [新增] AI 代理的配置 ---
# 使用代理API而非官方API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# OpenAI 代理配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key-for-proxy")
OPENAI_PROXY_PATH = os.getenv("OPENAI_PROXY_PATH", "/proxy/my-openai")
OPENAI_TARGET_URL = os.getenv("OPENAI_TARGET_URL", "https://api.chatanywhere.tech")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# DeepSeek 代理配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "dummy-key-for-proxy")
DEEPSEEK_PROXY_PATH = os.getenv("DEEPSEEK_PROXY_PATH", "/proxy/deepseek")
DEEPSEEK_TARGET_URL = os.getenv("DEEPSEEK_TARGET_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_ENDPOINT = os.getenv("DEEPSEEK_ENDPOINT", "https://api.deepseek.com/v1/chat/completions")

# Ali (Doubao) 代理配置
ALI_API_KEY = os.getenv("ALI_API_KEY", "dummy-key-for-proxy")
ALI_PROXY_PATH = os.getenv("ALI_PROXY_PATH", "/proxy/ali")
ALI_TARGET_URL = os.getenv("ALI_TARGET_URL", "https://www.dmxapi.cn")
ALI_MODEL = os.getenv("ALI_MODEL", "doubao-seed-1-6-250615")

# --- 模型映射配置 ---
MODEL_MAPPING = {
    'gemma': 'nomic-embed-text',
    'nomic': 'nomic-embed-text',
    'qwen': 'qwen3-embedding:0.6b',
}
DEFAULT_EMBEDDING_MODEL = 'qwen3-embedding:0.6b'

# --- 日志系统 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- 初始化 ---
app = Flask(__name__)
CORS(app)

# 确保 MILVUS_PORT 被正确读取
MILVUS_PORT_STR = os.getenv("MILVUS_PORT", "19530")
try:
    MILVUS_PORT = int(MILVUS_PORT_STR)
except ValueError:
    logging.warning(f"MILVUS_PORT 环境变量 '{MILVUS_PORT_STR}' 不是一个有效的端口号, 将使用默认值 19530。")
    MILVUS_PORT = 19530


# endregion

# region Milvus (Vector DB) Connection
# --- 关键修复：连接失败时不崩溃 ---
try: 
    logging.info(f"正在连接到 Milvus (Host: {MILVUS_HOST}, Port: {MILVUS_PORT})...") 
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT) 
    logging.info("成功连接到 Milvus。") 
except Exception as e: 
    logging.error(f"连接 Milvus 失败: {e}") 
    logging.warning("服务器将继续运行，但知识库功能不可用。") 

# endregion

# region Database and ORM setup
def get_model_for_collection(collection_name: str) -> str:
    for key, model_name in MODEL_MAPPING.items():
        if key in collection_name:
            logging.info(f"集合 '{collection_name}' 匹配到关键词 '{key}'，使用模型: '{model_name}'")
            return model_name
    logging.info(f"集合 '{collection_name}' 未匹配到任何关键词，使用默认模型: '{DEFAULT_EMBEDDING_MODEL}'")
    return DEFAULT_EMBEDDING_MODEL

def get_ollama_embedding(text: str, model_name: str):
    try:
        payload = {"model": model_name, "prompt": text}
        response = requests.post(OLLAMA_EMBED_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        if "embedding" not in response_data:
            raise ValueError(f"Ollama API 响应 (模型: {model_name}) 中缺少 'embedding' 字段。")
        return response_data["embedding"]
    except requests.exceptions.RequestException as e:
        logging.error(f"调用 Ollama API (模型: {model_name}) 失败: {e}")
        raise RuntimeError(f"无法连接到 Ollama 服务。")
    except Exception as e:
        logging.error(f"从 Ollama (模型: {model_name}) 获取嵌入时出错: {e}")
        raise

# --- [核心修正] 统一创建与旧 Schema 一致的集合 ---
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        return Collection(collection_name)
    logging.info(f"集合 '{collection_name}' 不存在，正在创建 (维度: {dim})...")
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="full_path", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description=f"知识库集合: {collection_name}")
    collection = Collection(name=collection_name, schema=schema)
    index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

def text_to_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- [核心修正] 统一数据处理逻辑 ---
def upsert_file_to_milvus(file_path: str, collection_name: str, model_name: str):
    filename = os.path.basename(file_path)
    try:
        collection = Collection(collection_name)
        collection.load()
        delete_expr = f"source_file == '{filename}'"
        collection.delete(delete_expr)
        logging.info(f"已从 '{collection_name}' 中删除 '{filename}' 的旧条目。")
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        chunks = text_to_chunks(content)
        if not chunks: 
            logging.info(f"文件 '{filename}' 为空，无需插入新数据。")
            return
        
        entities_to_insert = []
        logging.info(f"为 '{filename}' 的 {len(chunks)} 个文本块并发生成嵌入 (使用 {INGEST_WORKERS} 个工作线程)...")
        with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as executor:
            future_to_chunk = {executor.submit(get_ollama_embedding, chunk, model_name): (i, chunk) for i, chunk in enumerate(chunks)}
            for future in as_completed(future_to_chunk):
                try:
                    embedding = future.result()
                    i, chunk = future_to_chunk[future]
                    entity = {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "source_file": filename,
                        "chunk_index": i,
                        "full_path": file_path,
                        "embedding": embedding
                    }
                    entities_to_insert.append(entity)
                except Exception as e:
                    logging.error(f"为 '{filename}' 的一个文本块生成嵌入时失败: {e}")
        if entities_to_insert:
            collection.insert(entities_to_insert)
            collection.flush()
            logging.info(f"成功为 '{filename}' 插入 {len(entities_to_insert)} 个新条目。")
    except Exception as e:
        logging.error(f"处理文件 '{filename}' 时发生严重错误: {e}")
    finally:
        if 'collection' in locals():
            collection.release()

def process_file_delete(file_path, collection_name):
    filename = os.path.basename(file_path)
    logging.info(f"检测到文件删除: {filename}")
    try:
        collection = Collection(collection_name)
        collection.load()
        delete_expr = f"source_file == '{filename}'"
        collection.delete(delete_expr)
        logging.info(f"已从 '{collection_name}' 中删除 '{filename}' 的所有相关条目。")
    except Exception as e:
        logging.error(f"删除文件 '{filename}' 的条目时失败: {e}")
    finally:
        if 'collection' in locals():
            collection.release()

# --- 文件监控处理器 ---
class KnowledgeBaseEventHandler(FileSystemEventHandler):
    def __init__(self, collection_to_watch, model_name, base_dir=None):
        self.collection_to_watch = collection_to_watch
        self.model_name = model_name
        self.base_dir = base_dir or KNOWLEDGE_BASE_DIR
        self.watch_path = os.path.normpath(os.path.join(self.base_dir, self.collection_to_watch))
        logging.info(f"监控器已初始化，目标路径: {self.watch_path}")
    def process_if_relevant(self, event):
        if event.is_directory or not (event.src_path.endswith(".txt") or event.src_path.endswith(".md")): return
        event_dir = os.path.normpath(os.path.dirname(event.src_path))
        if event_dir != self.watch_path: return
        if event.event_type in ('created', 'modified'):
            upsert_file_to_milvus(event.src_path, self.collection_to_watch, self.model_name)
        elif event.event_type == 'deleted':
            process_file_delete(event.src_path, self.collection_to_watch)
        elif event.event_type == 'moved':
            process_file_delete(event.src_path, self.collection_to_watch)
            dest_dir = os.path.normpath(os.path.dirname(event.dest_path))
            if dest_dir == self.watch_path:
                upsert_file_to_milvus(event.dest_path, self.collection_to_watch, self.model_name)
    def on_created(self, event): self.process_if_relevant(event)
    def on_modified(self, event): self.process_if_relevant(event)
    def on_deleted(self, event): self.process_if_relevant(event)
    def on_moved(self, event): self.process_if_relevant(event)


# --- 数据导入逻辑 ---
def ingest_data():
    if not os.path.exists(KNOWLEDGE_BASE_DIR) or not os.path.isdir(KNOWLEDGE_BASE_DIR):
        logging.error(f"知识库根目录 '{KNOWLEDGE_BASE_DIR}' 不存在或不是一个目录。")