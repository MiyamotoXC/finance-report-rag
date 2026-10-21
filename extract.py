# -*- coding: utf-8 -*-
"""研报结构化抽取：从每篇研报抽取 行业/评级/核心观点/关注风险，便于横向对比。"""
import os
import json
import glob

import dashscope

import config

FIELDS = ["行业", "评级", "核心观点", "关注风险"]


def extract(text):
    dashscope.api_key = config.DASHSCOPE_API_KEY
    prompt = "从研报中提取以下字段，以 JSON 返回（字段名用中文）：" + "、".join(FIELDS) + "\n\n" + text[:4000]
    resp = dashscope.Generation.call(
        model=config.LLM_MODEL, prompt=prompt, temperature=0, result_format="json"
    )
    try:
        return json.loads(resp.output.text)
    except Exception:
        return {"raw": resp.output.text if resp.output else ""}


def run():
    out = {}
    for p in sorted(glob.glob(os.path.join(config.DATA_DIR, "*.md"))):
        with open(p, encoding="utf-8") as f:
            text = f.read()
        out[os.path.basename(p)[:-3]] = extract(text)
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    path = os.path.join(config.INDEX_DIR, "reports.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结构化研报已存至 {path}")


if __name__ == "__main__":
    run()
