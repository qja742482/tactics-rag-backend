# ⚽ Tactics-RAG: 레알 마드리드 전술 Q&A 백엔드 API & 웹 챗봇

> **FastAPI**, **ChromaDB**, **Google Gemini API**를 활용하여 2026-27 무리뉴 체제 레알 마드리드 전술 문서 데이터를 기반으로 질문에 답변을 제공하는 RAG(Retrieval-Augmented Generation) 백엔드 및 실시간 대화형 챗봇 서비스입니다.

---

## 📐 1. System Architecture

```text
[ User / Client ]
       │
       ▼ (HTTP GET / 또는 POST /ask-tactics)
┌───────────────────────────────────────────────────────────┐
│                    FastAPI Backend                        │
│                                                           │
│  1. 루트(/) 접속 시 단일 HTML/JS 웹 챗봇 화면 서빙        │
│  2. 질문 수신 및 검증 (300자 제한, top_k 1~10 제한)       │
│  3. Gemini Embedding (gemini-embedding-001) 변환          │
│  4. ChromaDB Vector Store 유사도 검색 (Cosine Distance)   │
│  5. Gemini LLM (gemini-3.5-flash-lite) 답변 생성          │
│  6. 3회 자동 재시도(Backoff) 및 파일 단위 출처 반환       │
└───────────────────────────────────────────────────────────┘
       │                               │
       ▼                               ▼
┌───────────────┐               ┌───────────────┐
│ Chroma Vector │               │ Google Gemini │
│   Database    │               │     APIs      │
└───────────────┘               └───────────────┘

🛠️ 2. Tech Stack

Backend: Python 3.12+, FastAPI, Pydantic, Uvicorn

Vector DB: ChromaDB (로컬 퍼시스턴트 저장소, HNSW Cosine Space)

LLM & Embedding: Google GenAI SDK (gemini-3.5-flash-lite, gemini-embedding-001)

Frontend: Vanilla HTML5 / CSS3 / JavaScript (FastAPI FileResponse 통합 서빙)

Deployment: Render (Cloud Web Service, 빌드 시 ingest.py 클린 빌드 파이프라인 적용)

Data Management: Markdown (.md) 기반 데이터 입수 스크립트 (ingest.py)

🚀 3. 핵심 개발 및 고도화 내역
① RAG 백엔드 파이프라인 및 환각 방지 (main.py)
사용자의 전술 관련 질문을 Vector DB에서 유사도 검색(top_k=6, DISTANCE_LIMIT=0.25 환경변수 제어)하여 팩트 기반 컨텍스트를 구성합니다.

엄격한 SYSTEM_PROMPT를 적용하여 수집된 참고 자료 내부의 지식만을 근거로 답변하도록 강제하고, 기준 거리를 벗어나거나 자료가 없을 경우 자의적인 답변(Hallucination) 대신 정해진 안내 문구를 반환합니다.

② API 예외 처리 및 안정성 강화
Google Gemini API 호출 시 발생할 수 있는 503(서버 혼잡) 및 429(Rate Limit) 에러에 대비하여 최대 3회 지수 백오프(Exponential Backoff) 재시도 로직과 빈 응답 방어 처리를 탑재했습니다.

③ 단일 페이지 웹 챗봇 UI 서빙 (index.html)
기존 Swagger UI(/docs) 중심의 개발자 환경을 넘어서, 루트 경로(/) 접속 시 직관적인 대화형 챗봇 화면을 제공합니다.

답변 내 근거 자료 번호 및 참고 마크다운 파일 출처(Sources)가 실시간으로 표기되어 시연성을 극대화했습니다.

④ 마크다운 기반 데이터 파이프라인 (ingest.py)
전술 분석 문서를 .md 파일 형태로 작성한 뒤 스크립트를 실행하면 문서의 출처 섹션을 제외한 본문만 추출하여 청킹(Chunking) 및 임베딩을 거쳐 최신 ChromaDB에 반영합니다.

⑤ Cloud 자동 배포 파이프라인 (Render)
GitHub 저장소와 Render 파이프라인을 연동하고 Build Command에 python ingest.py를 포함시켜, 코드 푸시 시 자동으로 최신 문서 기반 벡터 DB 구축 및 라이브 배포가 수행되는 환경을 구축했습니다.

💡 4. 프로젝트 핵심 포인트 (Portfolio Strengths)
직관적인 라이브 데모 시연성: 복잡한 프론트엔드 빌드 과정 없이 FastAPI의 FileResponse를 활용하여 완성도 높은 챗봇 UI를 통합 서빙합니다.

검증 가능한 파일 단위 출처 매핑: 청크(조각) 단위로 출처가 중복되던 문제를 파일 단위 중복 제거(Deduplication) 구조로 개선하여, LLM이 인용하는 번호와 UI 하단에 표기되는 .md 파일 목록을 완벽하게 일치시켰습니다.

실용적인 백엔드 안정성 설계: 질문 길이(300자) 및 검색 개수(top_k 1~10) 제한, 외부 LLM API 부하에 대응하는 재시도 로직, 거리 임계값 기반 환각 차단을 통해 실제 서비스 수준의 안정성을 확보했습니다.