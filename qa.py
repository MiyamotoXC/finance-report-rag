# -*- coding: utf-8 -*-
"""加载索引 -> 检索 -> LLM 生成（标注来源，区分事实与观点）。"""
import os
import sys
import json

import numpy as np
import dashscope

import config


def load():
    mat = np.load(os.path.join(config.INDEX_DIR, "emb.npy"))
    with open(os.path.join(config.INDEX_DIR, "meta.json"), encoding="utf-8") as f:
        recs = json.load(f)
    return mat, recs


def retrieve(mat, recs, query, top_k=config.TOP_K):
    dashscope.api_key = config.DASHSCOPE_API_KEY
    q = dashscope.TextEmbedding.call(model=config.EMBEDDING_MODEL, input=query)
    q = np.array(q.output["embeddings"][0]["embedding"], dtype="float32")
    sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
    order = np.argsort(-sims)[:top_k]
    return [(recs[i], float(sims[i])) for i in order]


def answer(query):
    mat, recs = load()
    hits = retrieve(mat, recs, query)
    ctx = "\n\n".join(f"[来源：{t}]\n{c}" for (t, c), _ in hits)
    prompt = (
        "你是金融研报分析助手，仅依据下方研报片段回答，区分事实与观点并标注来源；"
        "涉及评级/目标价须注明出处，未提及则说'研报中未提及'。\n\n"
        f"{ctx}\n\n问题：{query}"
    )
    dashscope.api_key = config.DASHSCOPE_API_KEY
    resp = dashscope.Generation.call(model=config.LLM_MODEL, prompt=prompt, temperature=0)
    print(resp.output.text if resp.output else "生成失败")
    print("\n来源：")
    for (t, c), s in hits:
        print(f" - {t}（{s:.3f}）")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "哪些行业被给出增持评级？"
    answer(q)
