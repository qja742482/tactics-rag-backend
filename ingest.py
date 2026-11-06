import os
import shutil
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY(또는 GOOGLE_API_KEY)가 .env에 없습니다.")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
DB_PATH = "./chroma_db"
DATA_DIR = Path("data/tactics")  # 여기에 .md / .txt 문서를 추가하면 같이 적재됨

client = genai.Client(api_key=API_KEY)

# 기본 내장 전술 텍스트 (실제 자료로 교체 권장, 기준 시즌/출처를 적어 두세요)
TACTICS_TEXT = """
[레알 마드리드 전술 분석 리포트]

1. 카를로 안첼로티 감독의 4-3-1-2 (다이아몬드) 포메이션
- 안첼로티는 주드 벨링엄을 공격형 미드필더(1번 자리)에 배치하여 공격의 핵으로 활용합니다.
- 벨링엄은 중앙과 측면을 자유롭게 오가며 침투하고, 박스 안 타격 능력을 극대화합니다.

2. 비니시우스 주니오르와 로드리고의 투톱 움직임
- 비니시우스는 좌측 측면뿐만 아니라 중앙으로 파고드는 인사이드 포워드 움직임을 가져갑니다.
- 로드리고는 우측 넓은 지역을 커버하며 상대 수비라인을 벌려주는 역할을 맡습니다.

3. 미드필더진의 기동성과 볼 순환
- 추아메니 또는 카마빙가는 수비형 미드필더로서 포백을 보호하고 역습을 차단합니다.
- 발베르데는 우측 하프스페이스를 끊임없이 오르내리며 공수 연결고리 역할을 수행합니다.
"""


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """문단 단위로 묶고, 앞 청크 끝부분을 겹쳐서 문맥 유지"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur)
            cur = cur[-overlap:] + "\n\n" + p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), 50):
        res = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors += [e.values for e in res.embeddings]
    return vectors


def collect_sources() -> dict[str, str]:
    sources = {"builtin_tactics.txt": TACTICS_TEXT}
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.rglob("*")):
            if f.is_file() and f.suffix in (".md", ".txt"):
                key = f.relative_to(DATA_DIR).as_posix()  # 예: real_madrid/음바페_역할.md
                sources[key] = f.read_text(encoding="utf-8")
    return sources


def main():
    # 임베딩 모델이 바뀌었을 수 있으므로 DB를 깨끗이 다시 만듭니다.
    shutil.rmtree(DB_PATH, ignore_errors=True)
    col = chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(
        "tactics", metadata={"hnsw:space": "cosine"}
    )

    for name, text in collect_sources().items():
        chunks = chunk_text(text)
        ids = [hashlib.md5(f"{name}-{i}-{c}".encode()).hexdigest() for i, c in enumerate(chunks)]
        team = name.split("/")[0] if "/" in name else "common"
        metas = [{"source": name, "team": team, "chunk": i} for i in range(len(chunks))]
        col.upsert(ids=ids, documents=chunks, embeddings=embed(chunks), metadatas=metas)
        print(f"✅ {name}: {len(chunks)}개 청크 저장")

    print("총 청크 수:", col.count())


if __name__ == "__main__":
    main()