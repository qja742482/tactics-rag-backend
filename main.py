import os
import time

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY(또는 GOOGLE_API_KEY)가 .env에 없습니다.")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
# 자료 유무를 가르는 코사인 거리 기준. 작을수록 엄격함.
# 서버 로그에 찍히는 "거리"를 보고 조정하세요. (.env / Render 환경변수 DISTANCE_LIMIT 으로도 바꿀 수 있음)
DISTANCE_LIMIT = float(os.getenv("DISTANCE_LIMIT", "0.25"))

client = genai.Client(api_key=API_KEY)
col = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(
    "tactics", metadata={"hnsw:space": "cosine"}
)

app = FastAPI(title="Tactics RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """너는 레알 마드리드 전술 분석가야.
반드시 아래 [참고 자료]에 있는 내용만 근거로 한국어로 답해.
- 자료에 답이 없으면 형식을 따르지 말고 "제공된 자료에서는 확인할 수 없습니다."라고 한 문장으로만 답해.
- 자료에 답이 있을 때만 (1) 핵심 요약 한 줄, (2) 전술적 설명, (3) 근거가 된 자료 번호 순서로 써.
- 자료 번호는 [참고 자료]에 적힌 번호를 그대로 인용해.
- 선수 이름, 포메이션, 수치는 자료에 적힌 그대로 사용해."""

REFUSAL_MESSAGE = "제공된 자료에서는 확인할 수 없습니다."


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=300)  # 너무 긴 질문 차단
    top_k: int = Field(default=6, ge=1, le=10)  # 여러 문서를 합쳐야 하는 질문을 위해 기본 6


@app.get("/", response_class=FileResponse)
def health():
    return FileResponse("index.html")


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

    # 기준값 조정용 로그: 답해야 할 질문과 거절해야 할 질문의 거리를 비교해 보세요.
    print(
        f"질문: {q.question} / 거리: {[round(d, 3) for d in res['distances'][0]]} "
        f"(기준 {DISTANCE_LIMIT})"
    )

    hits = [
        (d, m, s)
        for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        if s <= DISTANCE_LIMIT
    ]
    if not hits:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    # 자료 번호는 "파일 단위"로 매긴다.
    # (LLM이 인용하는 번호와 화면의 출처 목록 번호가 어긋나지 않게 하기 위함)
    file_no: dict[str, int] = {}
    best: dict[str, tuple[float, int]] = {}  # 파일별로 가장 가까운 (거리, 청크번호)
    for _, m, s in hits:
        src = m["source"]
        file_no.setdefault(src, len(file_no) + 1)
        if src not in best or s < best[src][0]:
            best[src] = (s, m["chunk"])

    context = "\n\n".join(
        f"[{file_no[m['source']]}] ({m['source']})\n{d}" for d, m, _ in hits
    )
    prompt = f"[참고 자료]\n{context}\n\n[질문]\n{q.question}"

    resp = None
    last_err = None
    for attempt in range(3):
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
            if ("503" in msg or "429" in msg) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            break

    if resp is None:
        raise HTTPException(status_code=502, detail=f"LLM 호출 실패: {last_err}")

    # resp.text 가 None 일 수 있어서(차단·빈 응답) 방어
    answer_text = (resp.text or "").strip()
    if not answer_text:
        raise HTTPException(
            status_code=502, detail="LLM이 빈 답변을 반환했습니다. 잠시 후 다시 시도하세요."
        )

    if REFUSAL_MESSAGE in answer_text:
        sources = []
    else:
        sources = [
            {
                "no": n,
                "file": src.split("/")[-1],  # 파일명만 (예: 벨링엄_역할.md)
                "full_path": src,
                "chunk": best[src][1],
                "distance": round(best[src][0], 3),
            }
            for src, n in file_no.items()
        ]

    um = resp.usage_metadata
    if um:
        print(
            f"토큰 사용: 입력 {um.prompt_token_count}, 출력 {um.candidates_token_count}, 합계 {um.total_token_count}"
        )

    return {"answer": answer_text, "sources": sources}