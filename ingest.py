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

# 검색에 넣지 않을 섹션 제목 (문서 맨 아래 출처 목록 등)
SKIP_SECTIONS = ("## 출처",)

client = genai.Client(api_key=API_KEY)


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


def prepare(text: str) -> tuple[str, str]:
    """문서에서 제목(첫 '# ' 줄)과 본문을 분리하고, 출처 같은 제외 섹션은 잘라낸다."""
    kept = []
    for line in text.splitlines():
        if line.strip().startswith(SKIP_SECTIONS):
            break  # 이 줄부터 문서 끝까지는 검색에 넣지 않음
        kept.append(line)
    body_lines = kept

    title = ""
    if body_lines and body_lines[0].startswith("# "):
        title = body_lines[0][2:].strip()
        body_lines = body_lines[1:]
    return title, "\n".join(body_lines).strip()


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
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} 폴더가 없습니다. 문서를 넣은 뒤 다시 실행하세요.")
    sources = {}
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file() and f.suffix in (".md", ".txt"):
            key = f.relative_to(DATA_DIR).as_posix()  # 예: real_madrid/player/음바페_역할.md
            sources[key] = f.read_text(encoding="utf-8")
    if not sources:
        raise ValueError(f"{DATA_DIR} 안에 .md / .txt 문서가 없습니다.")
    return sources


def main():
    # 임베딩 모델이 바뀌었을 수 있으므로 DB를 깨끗이 다시 만듭니다.
    shutil.rmtree(DB_PATH, ignore_errors=True)
    col = chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(
        "tactics", metadata={"hnsw:space": "cosine"}
    )

    for name, text in collect_sources().items():
        title, body = prepare(text)
        chunks = chunk_text(body)
        if not chunks:
            print(f"⚠️ {name}: 내용이 비어 있어 건너뜁니다")
            continue

        # 모든 청크 앞에 문서 제목을 붙여서 "누구/무엇에 대한 내용인지"를 검색이 알 수 있게 함
        if title:
            chunks = [f"[{title}]\n{c}" for c in chunks]

        ids = [hashlib.md5(f"{name}-{i}-{c}".encode()).hexdigest() for i, c in enumerate(chunks)]
        parts = name.split("/")
        team = parts[0] if len(parts) > 1 else "common"
        kind = parts[1] if len(parts) > 2 else "common"  # player / team
        metas = [
            {"source": name, "team": team, "kind": kind, "title": title, "chunk": i}
            for i in range(len(chunks))
        ]
        col.upsert(ids=ids, documents=chunks, embeddings=embed(chunks), metadatas=metas)
        print(f"✅ {name}: {len(chunks)}개 청크 저장")

    print("총 청크 수:", col.count())


if __name__ == "__main__":
    main()
