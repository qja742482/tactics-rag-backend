import os
import time

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY(또는 GOOGLE_API_KEY)가 .env에 없습니다.")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
DISTANCE_LIMIT = 0.35  # 자료 유무를 확실하게 가르기 위한 보정값  # cosine distance, 작을수록 유사. 응답의 distance를 보며 조정

client = genai.Client(api_key=API_KEY)
col = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(
    "tactics", metadata={"hnsw:space": "cosine"}
)

app = FastAPI(title="Tactics RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트 연동 시 실제 주소로 좁히세요
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """너는 레알 마드리드 전술 분석가야.
반드시 아래 [참고 자료]에 있는 내용만 근거로 한국어로 답해.
- 자료에 답이 없으면 형식을 따르지 말고 "제공된 자료에서는 확인할 수 없습니다."라고 한 문장으로만 답해.
- 자료에 답이 있을 때만 (1) 핵심 요약 한 줄, (2) 전술적 설명, (3) 근거가 된 자료 번호 순서로 써.
- 선수 이름, 포메이션, 수치는 자료에 적힌 그대로 사용해."""


class Question(BaseModel):
    question: str
    top_k: int = 4


@app.get("/")
def health():
    return {"status": "ok", "chunks": col.count()}


@app.post("/ask-tactics")
def ask_tactics(q: Question):
    if col.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="DB가 비어 있습니다. python ingest.py를 먼저 실행하세요.",
        )

    try:
        q_emb = (
            client.models.embed_content(
                model=EMBED_MODEL,
                contents=q.question,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            .embeddings[0]
            .values
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"임베딩 호출 실패: {e}")

    res = col.query(query_embeddings=[q_emb], n_results=q.top_k)
    hits = [
        (d, m, s)
        for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        if s <= DISTANCE_LIMIT
    ]
    if not hits:
        return {"answer": "관련된 전술 자료를 찾지 못했습니다.", "sources": []}

    context = "\n\n".join(
        f"[{i + 1}] ({m['source']})\n{d}" for i, (d, m, _) in enumerate(hits)
    )
    prompt = f"[참고 자료]\n{context}\n\n[질문]\n{q.question}"

    resp = None
    last_err = None
    for attempt in range(3):  # 최대 3번 시도
        try:
            resp = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            # 503(혼잡), 429(한도)만 재시도하고, 나머지는 바로 실패 처리
            if ("503" in msg or "429" in msg) and attempt < 2:
                time.sleep(2 * (attempt + 1))  # 2초, 4초 대기
                continue
            break

    if resp is None:
        raise HTTPException(status_code=502, detail=f"LLM 호출 실패: {last_err}")

    answer_text = resp.text.strip()

    # 1번 작업: 거절 답변인 경우 sources를 빈 배열로 비우기
    refusal_message = "제공된 자료에서는 확인할 수 없습니다."
    sources = []
    if refusal_message in answer_text:
        sources = []
    else:
        sources = [
            {
                "no": i + 1,
                "file": m["source"],
                "chunk": m["chunk"],
                "distance": round(s, 3),
            }
            for i, (_, m, s) in enumerate(hits)
        ]

    um = resp.usage_metadata
    if um:
        print(
            f"토큰 사용: 입력 {um.prompt_token_count}, 출력 {um.candidates_token_count}, 합계 {um.total_token_count}"
        )

    return {"answer": answer_text, "sources": sources}