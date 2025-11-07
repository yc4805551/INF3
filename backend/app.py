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

MILVUS_PORT = os.getenv("MILVUS_PORT")

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
        return
    for collection_name in os.listdir(KNOWLEDGE_BASE_DIR):
        collection_path = os.path.join(KNOWLEDGE_BASE_DIR, collection_name)
        if not os.path.isdir(collection_path): continue
        logging.info(f"\n--- 正在处理目录 (集合): {collection_name} ---")
        model_to_use = get_model_for_collection(collection_name)
        try:
            logging.info(f"正在检测模型 '{model_to_use}' 的向量维度...")
            dummy_embedding = get_ollama_embedding("test", model_to_use)
            dim = len(dummy_embedding)
            logging.info(f"检测到维度为: {dim}")
        except Exception as e:
            logging.error(f"无法为模型 '{model_to_use}' 获取向量维度，跳过此目录。错误: {e}")
            continue
        create_milvus_collection(collection_name, dim)
        for filename in os.listdir(collection_path):
            file_path = os.path.join(collection_path, filename)
            if not (filename.endswith(".txt") or filename.endswith(".md")): continue
            logging.info(f"开始处理文件: {filename}")
            upsert_file_to_milvus(file_path, collection_name, model_to_use)

# --- Flask CLI 命令 ---
@app.cli.command("ingest")
def ingest_command():
    ingest_data()
    click.echo("数据导入过程完成。")
@app.cli.command("watch")
def watch_command():
    collection_to_watch = 'kb_qwen_0_6b'
    model_name = get_model_for_collection(collection_to_watch)
    if not utility.has_collection(collection_to_watch):
        click.echo(f"错误: 目标知识库 '{collection_to_watch}' 在 Milvus 中不存在。")
        click.echo(f"请先运行 'flask ingest' 来创建和初始化所有知识库。")
        return
    path_to_watch = KNOWLEDGE_BASE_DIR
    event_handler = KnowledgeBaseEventHandler(collection_to_watch, model_name)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    click.echo(f"✅ 已启动监控服务，只处理对 '{collection_to_watch}' 知识库的更新。")
    click.echo(f"   监控目录: {os.path.join(path_to_watch, collection_to_watch)}")
    click.echo("   按 Ctrl+C 停止服务。")
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    click.echo("\n监控服务已停止。")
@app.cli.command("watch-nomic")
def watch_nomic_command():
    collection_to_watch = 'kb_nomic'
    model_name = get_model_for_collection(collection_to_watch)
    if not utility.has_collection(collection_to_watch):
        click.echo(f"错误: 目标知识库 '{collection_to_watch}' 在 Milvus 中不存在。")
        click.echo(f"请先运行 'flask ingest' 来创建和初始化所有知识库。")
        return
    path_to_watch = KNOWLEDGE_BASE_DIR_NOMIC
    event_handler = KnowledgeBaseEventHandler(collection_to_watch, model_name, base_dir=KNOWLEDGE_BASE_DIR_NOMIC)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    click.echo(f"✅ 已启动监控服务，只处理对 '{collection_to_watch}' 知识库的更新。")
    click.echo(f"   监控目录: {os.path.join(path_to_watch, collection_to_watch)}")
    click.echo("   按 Ctrl+C 停止服务。")
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    click.echo("\n监控服务已停止。")


# --- 知识库 API 端点 ---
@app.route('/api/list-collections', methods=['GET'])
def list_collections():
    try:
        # 检查并重新建立Milvus连接
        try:
            # 尝试获取集合列表，如果失败则重新连接
            collections = utility.list_collections()
        except Exception as conn_error:
            logging.warning(f"Milvus连接可能已断开，尝试重新连接: {conn_error}")
            # 重新建立连接
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            collections = utility.list_collections()
            logging.info("Milvus连接已重新建立")
        
        return jsonify({"collections": collections})
    except Exception as e:
        logging.error(f"API /list-collections 失败: {e}")
        return jsonify({"error": "无法获取 Milvus 集合列表", "details": str(e)}), 500

@app.route('/api/find-related', methods=['POST'])
def find_related():
    try:
        data = request.get_json()
        query_text = data.get('text')
        collection_name = data.get('collection_name')
        top_k = data.get('top_k', 10)
        if not query_text or not collection_name:
            return jsonify({"error": "请求中缺少 'text' 或 'collection_name'"}), 400
        
        # 检查并重新建立Milvus连接
        try:
            if not utility.has_collection(collection_name):
                return jsonify({"error": f"知识库 (集合) '{collection_name}' 不存在。"}), 404
        except Exception as conn_error:
            logging.warning(f"Milvus连接可能已断开，尝试重新连接: {conn_error}")
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
            logging.info("Milvus连接已重新建立")
            if not utility.has_collection(collection_name):
                return jsonify({"error": f"知识库 (集合) '{collection_name}' 不存在。"}), 404

        model_to_use = get_model_for_collection(collection_name)
        query_embedding = get_ollama_embedding(query_text, model_to_use)
        collection = Collection(collection_name)
        collection.load()
        schema_fields = {field.name: field for field in collection.schema.fields}
        output_fields = ["text", "source_file", "chunk_index", "full_path"]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=top_k, output_fields=output_fields)
        response_data = []
        for hit in results[0]:
            entity = hit.entity
            response_data.append({
                "source_file": entity.get("source_file", "Unknown Source"),
                "content_chunk": entity.get("text", ""),
                "score": hit.distance,
            })
        collection.release()
        return jsonify({"related_documents": response_data})
    except (RuntimeError, ValueError) as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logging.error(f"API /find-related 发生内部错误: {e}", exc_info=True)
        return jsonify({"error": "服务器内部错误", "details": str(e)}), 500

@app.route('/api/', methods=['GET'])
def index():
    return "知识库后端 + AI 代理服务器正在运行 (v13 + 代理)。"


# --- [新增] AI 代理端点 ---

def _format_history(history):
    """辅助函数：格式化历史记录"""
    # (这里您可以根据不同模型的需要调整格式)
    return history

# region AI Generation
@app.route('/api/generate', methods=['POST']) 
def handle_generate(): 
    """处理非流式 AI 生成请求 (v2 - 增加 Gemini JSON 清理)""" 
    try: 
        data = request.get_json() 
        provider = data.get('provider') 
        
        logging.info(f"收到非流式生成请求，Provider: {provider}") 
        
        full_response = "" 
        
        if provider == 'gemini': 
            if not GEMINI_API_KEY: 
                return jsonify({"error": "GEMINI_API_KEY 未设置"}), 500 
            
            for chunk in _stream_gemini(data.get('userPrompt'), data.get('systemInstruction'), data.get('history', [])): 
                if "[后端" in chunk: 
                    raise Exception(chunk) 
                full_response += chunk 
            
            # --- ⬇️ 关键修复：清理 Gemini 返回的损坏的 JSON ⬇️ --- 
            logging.info(f"Gemini 原始响应: {full_response}") 
            
            json_string = full_response 
            
            # 1. 从Markdown中提取
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', full_response, re.IGNORECASE) 
            if json_match: 
                json_string = json_match.group(1) 

            # 2. 修复 Gemini 生成的无效JSON键 (例如 \"key\":) 
            # 这会将 \"problematicText\": 替换为 "problematicText": 
            cleaned_string = re.sub(r'\\"(\w+)\\"(\s*):', r'"\1"\2:', json_string) 
            
            # 3. 修复 Gemini 生成的无效JSON字符串 (例如 \"value\") 
            # 这会将 \"测1试\\n\" 替换为 "测1试\\n" 
            cleaned_string = re.sub(r':\s*\\"(.*?)\\"', r': "\1"', cleaned_string) 
            
            logging.info(f"清理后的 JSON: {cleaned_string}") 
            return Response(cleaned_string, content_type='application/json') 
            # --- ⬆️ 清理结束 ⬆️ --- 
            
        elif provider == 'openai': 
            return _call_openai_proxy(data) 
        elif provider == 'deepseek': 
            return _call_deepseek_proxy(data) 
        elif provider == 'ali': 
            return _call_ali_proxy(data) 
        else: 
            return jsonify({"error": f"不支持的 provider: {provider}"}), 400 

    except Exception as e: 
        logging.error(f"API /api/generate 错误: {e}") 
        return jsonify({"error": "服务器内部错误", "details": str(e)}), 500 

@app.route('/api/generate-stream', methods=['POST']) 
def handle_generate_stream(): 
    """处理流式 AI 生成请求""" 
    try: 
        data = request.get_json() 
        provider = data.get('provider') 
        system_instruction = data.get('systemInstruction') 
        user_prompt = data.get('userPrompt') 
        history = data.get('history', []) 
        
        logging.info(f"收到 /api/generate-stream 请求，Provider: {provider}") 

        if provider == 'gemini': 
            if not GEMINI_API_KEY: 
                # 这个错误应该在流中返回，而不是作为JSON 
                return Response(stream_with_context(["[后端代理错误: GEMINI_API_KEY 未设置]"]), content_type='text/plain') 
            return Response(stream_with_context(_stream_gemini(user_prompt, system_instruction, history)), content_type='text/plain') 
        
        elif provider == 'openai': 
            if not OPENAI_API_KEY: 
                return Response(stream_with_context(["[后端代理错误: OPENAI_API_KEY 未设置]"]), content_type='text/plain') 
            return Response(stream_with_context(_stream_openai_proxy(user_prompt, system_instruction, history)), content_type='text/plain') 
            
        elif provider == 'deepseek': 
            if not DEEPSEEK_API_KEY: 
                return Response(stream_with_context(["[后端代理错误: DEEPSEEK_API_KEY 未设置]"]), content_type='text/plain') 
            return Response(stream_with_context(_stream_deepseek_proxy(user_prompt, system_instruction, history)), content_type='text/plain') 
            
        elif provider == 'ali': 
            if not ALI_API_KEY: 
                return Response(stream_with_context(["[后端代理错误: ALI_API_KEY 未设置]"]), content_type='text/plain') 
            return Response(stream_with_context(_stream_ali_proxy(user_prompt, system_instruction, history)), content_type='text/plain') 

        else: 
            return Response(stream_with_context([f"[后端代理错误: 不支持的 provider: {provider}]"]), content_type='text/plain') 

    except Exception as e: 
        logging.error(f"API /api/generate-stream 错误: {e}") 
        return Response(stream_with_context([f"[后端内部错误: {str(e)}]"]), content_type='text/plain')

def _stream_gemini(user_prompt, system_instruction, history): 
    """处理Gemini模型的流式响应 (v3 - 修复 try/except 逻辑)""" 
    try: 
        # 初始化变量 
        contents = [] 
        last_role = None # 初始 last_role 必须是 None 

        # 1. 处理 System Instruction (如果存在) 
        if system_instruction: 
            contents.append({"role": "user", "parts": [{"text": system_instruction}]}) 
            contents.append({"role": "model", "parts": [{"text": "好的，我将遵循这个指示。"}]}) 
            last_role = "model" 
        
        # 2. 处理历史消息 
        for item in history: 
            current_role = item.get('role') 
            if not current_role: 
                logging.warning(f"跳过缺少角色的历史记录: {item}") 
                continue 
            
            # 验证parts格式 
            parts = item.get('parts') 
            if not parts or not isinstance(parts, list) or len(parts) == 0 or not parts[0].get('text'): 
                logging.warning(f"跳过格式无效的历史记录: {item}") 
                continue 
            
            # 核心修复：如果 'contents' 为空，第一个角色必须是 'user' 
            if not contents and current_role == 'model': 
                logging.warning("跳过历史记录中开头的 'model' 消息。") 
                continue 
                
            # 核心修复：如果当前角色与上一个角色相同，跳过此条 
            if current_role == last_role: 
                logging.warning(f"跳过重复的角色: {current_role}") 
                continue 
            
            # 添加有效的历史消息 
            contents.append(item) 
            last_role = current_role 
        
        # 3. 添加当前用户的消息 
        if last_role == 'user': 
            # 如果历史的最后一条是'user'，添加一个假的'model'回复 
            contents.append({"role": "model", "parts": [{"text": "..."}]}) 
        
        contents.append({"role": "user", "parts": [{"text": user_prompt}]}) 
        
        # 4. 构建API请求 
        # --- 关键修复：使用 gemini-pro 而不是 1.5-flash ---
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?key={GEMINI_API_KEY}" 
        headers = {'Content-Type': 'application/json'} 
        payload = { 
            "contents": contents, 
            "generationConfig": { 
                "temperature": 0.7, 
                "maxOutputTokens": 2048 
            } 
        } 
        
        # --- 关键修复：注释掉不兼容的 systemInstruction 字段 ---
        # if system_instruction: 
        #     payload["systemInstruction"] = {"parts": [{"text": system_instruction}]} 
        
        # 5. 发送API请求 
        logging.info(f"发送到 Gemini 的 Payload (精简版): {len(contents)} 条消息。") 
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r: 
            
            if not r.ok: 
                error_details = r.text 
                logging.error(f"Gemini API 请求失败 (状态码: {r.status_code}): {error_details}") 
                yield f"[后端代理错误: Gemini API 返回 {r.status_code}. 详情: {error_details}]" 
                return 

            # 6. 处理流式响应 
            for line in r.iter_lines(): 
                if line: 
                    line_str = line.decode('utf-8').strip() 
                    if line_str.startswith('"text":'): 
                        text_chunk = line_str.replace('"text": "', '').replace('"', '').replace(',', '').strip() 
                        yield text_chunk 

    except requests.exceptions.RequestException as e: # 捕获网络请求错误 
        logging.error(f"调用 Gemini API 失败: {e}") 
        yield f"[后端代理错误: {str(e)}]" 
    except Exception as e: # 捕获所有其他错误 
        logging.error(f"处理 Gemini 流时出错: {e}") 
        yield f"[后端内部错误: {str(e)}]" 

def _call_openai_proxy(data):
    """通过代理调用OpenAI兼容的API（非流式）"""
    try:
        url = f"{OPENAI_TARGET_URL}/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        
        # 构建请求体
        messages = []
        if data.get('systemInstruction'):
            messages.append({"role": "system", "content": data.get('systemInstruction')})
        
        # 添加历史消息
        for item in data.get('history', []):
            if item.get('role') and item.get('parts') and len(item.get('parts')) > 0:
                messages.append({
                    "role": item.get('role'),
                    "content": item.get('parts')[0].get('text')
                })
        
        # 添加用户消息
        messages.append({"role": "user", "content": data.get('userPrompt')})
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            # --- 关键修复：返回原始 Response，而非 jsonify ---
            return Response(response_data['choices'][0]['message']['content'], content_type='application/json')
        else:
            raise ValueError("Invalid response format from OpenAI proxy")
    
    except Exception as e:
        logging.error(f"调用 OpenAI 代理失败: {e}")
        raise

def _stream_openai_proxy(user_prompt, system_instruction, history):
    """流式调用代理的 OpenAI API"""
    try:
        url = f"{OPENAI_TARGET_URL}/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        
        # 构建请求体
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        # 添加历史消息
        for item in history:
            if item.get('role') and item.get('parts') and len(item.get('parts')) > 0:
                messages.append({
                    "role": item.get('role'),
                    "content": item.get('parts')[0].get('text')
                })
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_prompt})
        
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "stream": True
        }
        
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                        if line_str == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(line_str)
                            if ('choices' in chunk_data and 
                                len(chunk_data['choices']) > 0 and
                                chunk_data['choices'][0].get('delta') and
                                'content' in chunk_data['choices'][0]['delta']):
                                content = chunk_data['choices'][0]['delta']['content']
                                yield content
                        except json.JSONDecodeError:
                            logging.warning(f"无法解析 OpenAI 流式响应: {line_str}")
    
    except requests.exceptions.RequestException as e:
        logging.error(f"调用 OpenAI 代理失败: {e}")
        yield f"[后端代理错误: {str(e)}]"
    except Exception as e:
        logging.error(f"处理 OpenAI 流时出错: {e}")
        yield f"[后端内部错误: {str(e)}]"

def _call_deepseek_proxy(data):
    """调用代理的 DeepSeek API"""
    try:
        url = DEEPSEEK_ENDPOINT
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
        }
        
        # 构建请求体
        messages = []
        if data.get('systemInstruction'):
            messages.append({"role": "system", "content": data.get('systemInstruction')})
        
        # 添加历史消息
        for item in data.get('history', []):
            if item.get('role') and item.get('parts') and len(item.get('parts')) > 0:
                messages.append({
                    "role": item.get('role'),
                    "content": item.get('parts')[0].get('text')
                })
        
        # 添加用户消息
        messages.append({"role": "user", "content": data.get('userPrompt')})
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            # --- 关键修复：返回原始 Response，而非 jsonify ---
            return Response(response_data['choices'][0]['message']['content'], content_type='application/json')
        else:
            raise ValueError("Invalid response format from DeepSeek proxy")
    
    except Exception as e:
        logging.error(f"调用 DeepSeek 代理失败: {e}")
        raise

def _stream_deepseek_proxy(user_prompt, system_instruction, history):
    """流式调用代理的 DeepSeek API"""
    try:
        url = DEEPSEEK_ENDPOINT
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
        }
        
        # 构建请求体
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        # 添加历史消息
        for item in history:
            if item.get('role') and item.get('parts') and len(item.get('parts')) > 0:
                messages.append({
                    "role": item.get('role'),
                    "content": item.get('parts')[0].get('text')
                })
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_prompt})
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "stream": True
        }
        
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                        if line_str == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(line_str)
                            if ('choices' in chunk_data and 
                                len(chunk_data['choices']) > 0 and
                                chunk_data['choices'][0].get('delta') and
                                'content' in chunk_data['choices'][0]['delta']):
                                content = chunk_data['choices'][0]['delta']['content']
                                yield content
                        except json.JSONDecodeError:
                            logging.warning(f"无法解析 DeepSeek 流式响应: {line_str}")
    
    except requests.exceptions.RequestException as e:
        logging.error(f"调用 DeepSeek 代理失败: {e}")
        yield f"[后端代理错误: {str(e)}]"
    except Exception as e:
        logging.error(f"处理 DeepSeek 流时出错: {e}")
        yield f"[后端内部错误: {str(e)}]"

def _call_ali_proxy(data):
    """调用代理的 Ali (Doubao) API"""
    try:
        # 阿里云通义千问的 API 格式可能有所不同
        url = f"{ALI_TARGET_URL}/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {ALI_API_KEY}'
        }
        
        # 构建请求体
        messages = []
        if data.get('systemInstruction'):
            messages.append({"role": "system", "content": data.get('systemInstruction')})
        
        # 添加历史消息
        for item in data.get('history', []):
            if item.get('role') and item.get('parts') and len(item.get('parts')) > 0:
                messages.append({
                    "role": item.get('role'),
                    "content": item.get('parts')[0].get('text')
                })
        
        # 添加用户消息
        messages.append({"role": "user", "content": data.get('userPrompt')})
        
        payload = {
            "model": ALI_MODEL,
            "messages": messages,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        
        if 'choices' in response_data and len(response_data['choices']) > 0:
            # --- 关键修复：返回原始 Response，而非 jsonify ---
            return Response(response_data['choices'][0]['message']['content'], content_type='application/json')
        else:
            raise ValueError("Invalid response format from Ali proxy")
    
    except Exception as e:
        logging.error(f"调用 Ali 代理失败: {e}")
        raise

def _stream_ali_proxy(user_prompt, system_instruction, history):
    """流式调用代理的 Ali (Doubao) API"""
    try:
        # 阿里云通义千问的 API 格式可能有所不同
        url = f"{ALI_TARGET_URL}/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {ALI_API_KEY}'
        }
        
        # 构建请求体
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        
        # 添加历史消息
        for item in history:
            if item.get('role') and item.get('parts') and len(item.get('parts')) > 0:
                messages.append({
                    "role": item.get('role'),
                    "content": item.get('parts')[0].get('text')
                })
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_prompt})
        
        payload = {
            "model": ALI_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "stream": True
        }
        
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        line_str = line_str[6:]
                        if line_str == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(line_str)
                            if ('choices' in chunk_data and 
                                len(chunk_data['choices']) > 0 and
                                chunk_data['choices'][0].get('delta') and
                                'content' in chunk_data['choices'][0]['delta']):
                                content = chunk_data['choices'][0]['delta']['content']
                                yield content
                        except json.JSONDecodeError:
                            logging.warning(f"无法解析 Ali 流式响应: {line_str}")
    
    except requests.exceptions.RequestException as e:
        logging.error(f"调用 Ali 代理失败: {e}")
        yield f"[后端代理错误: {str(e)}]"
    except Exception as e:
        logging.error(f"处理 Ali 流时出错: {e}")
        yield f"[后端内部错误: {str(e)}]"


if __name__ == '__main__':
    # 注意: 环境变量 FLASK_APP=app.py
    # 运行: flask run --port=5000
    # (或在生产环境中使用 gunicorn)
    app.run(debug=True, port=5000)