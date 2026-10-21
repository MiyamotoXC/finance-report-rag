# -*- coding: utf-8 -*-
"""金融研报 RAG：统一配置（密钥走环境变量，不入库）。"""
import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误：请设置 'DASHSCOPE_API_KEY' 环境变量。")

EMBEDDING_MODEL = "text-embedding-v1"
LLM_MODEL = "qwen-plus"

DATA_DIR = "example_reports"
INDEX_DIR = "vector_store"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
TOP_K = 4
