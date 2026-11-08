# ⚽ Tactics-RAG: 레알 마드리드 전술 Q&A 백엔드 API

> **FastAPI**, **ChromaDB**, **Google Gemini API**를 활용하여 레알 마드리드 전술 문서 데이터를 기반으로 질문에 답변을 제공하는 RAG(Retrieval-Augmented Generation) 백엔드 서비스입니다.

---

## 📐 1. System Architecture

```text
[ User / Client ]
       │
       ▼ (HTTP POST /ask-tactics)
┌───────────────────────────────────────────────────────────┐
│                      FastAPI Backend                      │
│                                                           │
│  1. 질문 수신 (Question Model)                              │
│  2. Gemini Embedding (gemini-embedding-001) 변환            │
│  3. ChromaDB Vector Store 유사도 검색 (Cosine Distance)    │
│  4. Gemini LLM (gemini-3.5-flash-lite) 답변 생성          │
│  5. 예외 처리 & 출처(Sources) 포함 JSON 응답 반환            │
└───────────────────────────────────────────────────────────┘
       │                              │
       ▼                              ▼
┌───────────────┐              ┌───────────────┐
│  Chroma Vector│              │ Google Gemini │
│   Database    │              │     APIs      │
└───────────────┘              └───────────────┘