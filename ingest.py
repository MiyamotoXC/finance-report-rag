# -*- coding: utf-8 -*-
"""解析研报 -> 切分 -> 云端向量化 -> numpy 索引落盘。"""
import os
import glob
import json

import numpy as np
import dashscope

import config


def load_reports(data_dir=config.DATA_DIR):
    out = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(p, encoding="utf-8") as f:
            text = f.read()
        out.append((os.path.basename(p)[:-3], text))
    return out


def split_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) <= size:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > size:
                for i in range(0, len(p), size - overlap):
                    chunks.append(p[i : i + size])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def embed_batch(texts):
    resp = dashscope.TextEmbedding.call(model=config.EMBEDDING_MODEL, input=texts)
    return np.array([d["embedding"] for d in resp.output["embeddings"]], dtype="float32")


def build():
    dashscope.api_key = config.DASHSCOPE_API_KEY
    records = []
    for title, text in load_reports():
        for c in split_text(text):
            records.append((title, c))
    texts = [r[1] for r in records]

    mat = np.vstack([embed_batch(texts[i : i + 16]) for i in range(0, len(texts), 16)])

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    np.save(os.path.join(config.INDEX_DIR, "emb.npy"), mat)
    with open(os.path.join(config.INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"已索引 {len(records)} 个切片，存至 {config.INDEX_DIR}")


if __name__ == "__main__":
    build()
