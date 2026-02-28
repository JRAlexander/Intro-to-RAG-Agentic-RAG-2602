from __future__ import annotations

import json
from typing import Any

import pandas as pd
from openai import OpenAI

RUBRIC = """You are grading answer helpfulness for an internal notes QA task.
Return strict JSON with keys:
- score: integer 1-5
- rationale: short string
Rubric:
1 = unhelpful or clearly wrong
3 = partially helpful but missing key context
5 = correct, concise, and appropriately caveated
"""


def judge_answer(question: str, answer: str, model: str = "gpt-4o-mini") -> dict[str, Any]:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
        ],
    )
    parsed = json.loads(response.choices[0].message.content or "{}")
    return {"llm_judge_score": int(parsed.get("score", 0)), "llm_judge_rationale": parsed.get("rationale", "")}


def llm_judge_dataframe(df: pd.DataFrame, model: str = "gpt-4o-mini") -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        verdict = judge_answer(question=str(row["question"]), answer=str(row["answer"]), model=model)
        rows.append(verdict)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
