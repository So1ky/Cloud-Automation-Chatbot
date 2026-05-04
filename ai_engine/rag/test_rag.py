"""RAG 전체 파이프라인 품질 테스트 스크립트."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from ai_engine.rag.knowledge_base import (
    load_knowledge_base,
    translate_query,
    generate_multi_queries,
    _get_bm25_retriever,
    rerank_documents,
)
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

queries = [
    ("비용 최적화", "EC2 비용 절감 및 Reserved Instance 전략"),
    ("보안",        "RDS 데이터베이스 보안 설정 및 암호화"),
    ("고가용성",    "Multi-AZ 배포로 고가용성 웹 서비스 구성"),
]

vectorstore = load_knowledge_base()
bm25_retriever = _get_bm25_retriever(k=8)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

for category, q in queries:
    print(f"\n{'='*60}")
    print(f"[{category}] 원본: {q}")

    translated = translate_query(q)
    print(f"[번역] {translated}")

    multi_queries = generate_multi_queries(q)
    print(f"[Multi-query] {len(multi_queries)}개 생성:")
    for i, mq in enumerate(multi_queries):
        print(f"  {i+1}. {mq}")

    # 하이브리드 검색
    all_queries = [translated] + multi_queries
    seen: set = set()
    candidates: list[Document] = []
    for mq in all_queries:
        vr = vectorstore.as_retriever(search_kwargs={"k": 8})
        bm25_retriever.k = 8
        ensemble = EnsembleRetriever(
            retrievers=[vr, bm25_retriever], weights=[0.6, 0.4]
        )
        for doc in ensemble.invoke(mq):
            key = doc.page_content[:150]
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

    # Reranker
    pairs = [(translated, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    print(f"\n[결과] 후보 {len(candidates)}개 → 상위 5개")
    print('='*60)
    for i, (score, doc) in enumerate(scored[:5]):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "").split("/")[-1]
        pillar = doc.metadata.get("pillar", "?")
        bp = doc.metadata.get("bp_numbers", "")
        services = doc.metadata.get("aws_services", "")
        print(f"\n[청크 {i+1}] score={score:.3f} | p.{page} | {source}")
        print(f"         pillar={pillar} | BP={bp or 'N/A'} | services={services or 'N/A'}")
        print(doc.page_content[:300])
