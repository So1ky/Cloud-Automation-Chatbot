"""AWS Well-Architected Framework PDF 문서를 ChromaDB에 인덱싱하고 검색하는 모듈.

개선 사항:
1. BP 섹션 경계 인식 스마트 청킹 + pillar/BP번호/AWS서비스 메타데이터
2. Multi-query 생성 (요구사항 → 4개 검색 쿼리)
3. 하이브리드 검색 (벡터 60% + BM25 40%)
4. Cross-encoder Reranker로 최종 정밀 재점수화
"""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever

AI_ENGINE_DIR = Path(__file__).parent.parent
DOCS_DIR = AI_ENGINE_DIR / "docs"
CHROMA_PERSIST_DIR = AI_ENGINE_DIR / "chroma_db"
COLLECTION_NAME = "well_architected"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# AWS 서비스 목록 (메타데이터 추출용)
AWS_SERVICES = [
    "EC2", "RDS", "S3", "VPC", "ALB", "NLB", "ELB", "Lambda", "ECS", "EKS",
    "CloudFront", "Route 53", "IAM", "KMS", "CloudWatch", "SNS", "SQS",
    "DynamoDB", "Aurora", "EBS", "EFS", "Secrets Manager", "CloudHSM",
    "WAF", "Shield", "GuardDuty", "Config", "CloudTrail", "Auto Scaling",
    "Elastic Beanstalk", "CodeDeploy", "CodePipeline", "Fargate",
]

# BP 번호 패턴 (예: SEC01-BP01, REL10-BP03)
BP_PATTERN = re.compile(r"\b[A-Z]{2,5}\d{1,2}-BP\d{2,3}\b")

# 세션 내 캐시
_vectorstore_cache: Chroma | None = None
_bm25_retriever_cache: BM25Retriever | None = None
_reranker_cache = None


# ─── 메타데이터 추출 헬퍼 ────────────────────────────────────────────────────

def _extract_pillar(filename: str) -> str:
    """파일명에서 Well-Architected Pillar 이름 추출."""
    match = re.search(r"wellarchitected-(.+?)(?:-pillar)?\.pdf", filename)
    if match:
        return match.group(1).replace("-", "_")
    return "framework"


def _extract_services(text: str) -> str:
    """청크 텍스트에서 언급된 AWS 서비스 목록을 콤마 구분 문자열로 반환."""
    found = [s for s in AWS_SERVICES if s in text]
    return ",".join(found)


def _smart_split(documents: list[Document]) -> list[Document]:
    """BP 섹션 경계를 존중하는 스마트 청킹 + 메타데이터 보강."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for doc in documents:
        filename = Path(doc.metadata.get("source", "")).name
        pillar = _extract_pillar(filename)

        sub_chunks = splitter.split_documents([doc])
        for chunk in sub_chunks:
            bp_numbers = BP_PATTERN.findall(chunk.page_content)
            services = _extract_services(chunk.page_content)
            chunk.metadata.update({
                "pillar": pillar,
                "bp_numbers": ",".join(bp_numbers),
                "aws_services": services,
            })
            chunks.append(chunk)
    return chunks


# ─── 인덱싱 ──────────────────────────────────────────────────────────────────

def build_knowledge_base() -> Chroma:
    """docs/ 폴더의 PDF를 로드해 ChromaDB에 임베딩 후 저장한다."""
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"ai_engine/docs/ 폴더에 PDF 파일이 없습니다. ({DOCS_DIR})"
        )

    documents = []
    for pdf_path in pdf_files:
        print(f"[RAG] 로딩 중: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    chunks = _smart_split(documents)
    print(f"[RAG] 총 {len(chunks)}개 청크 생성 완료")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )
    print(f"[RAG] ChromaDB 저장 완료: {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_knowledge_base() -> Chroma:
    """저장된 ChromaDB를 로드한다. 없거나 비어있으면 새로 빌드한다."""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if CHROMA_PERSIST_DIR.exists():
        db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
        count = db._collection.count()
        if count > 0:
            print(f"[RAG] 기존 ChromaDB 로드 중... ({count}개 청크)")
            _vectorstore_cache = db
            return _vectorstore_cache
        print("[RAG] ChromaDB가 비어 있습니다. 새로 빌드합니다...")

    _vectorstore_cache = build_knowledge_base()
    return _vectorstore_cache


# ─── 검색 헬퍼 ───────────────────────────────────────────────────────────────

def _get_bm25_retriever(k: int = 8) -> BM25Retriever:
    """BM25 인덱스를 구축한다. 세션 내 캐싱."""
    global _bm25_retriever_cache
    if _bm25_retriever_cache is not None:
        _bm25_retriever_cache.k = k
        return _bm25_retriever_cache

    print("[RAG] BM25 인덱스 구축 중...")
    vectorstore = load_knowledge_base()
    result = vectorstore.get()
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]
    _bm25_retriever_cache = BM25Retriever.from_documents(docs, k=k)
    print(f"[RAG] BM25 인덱스 완료 ({len(docs)}개 문서)")
    return _bm25_retriever_cache


def _get_reranker():
    """Cross-encoder reranker를 로드한다. 세션 내 캐싱."""
    global _reranker_cache
    if _reranker_cache is None:
        from sentence_transformers import CrossEncoder
        print(f"[RAG] Reranker 로드 중: {RERANKER_MODEL}")
        _reranker_cache = CrossEncoder(RERANKER_MODEL)
        print("[RAG] Reranker 로드 완료")
    return _reranker_cache


def rerank_documents(query: str, docs: list, top_k: int = 5) -> list:
    """Cross-encoder로 재점수화 후 상위 top_k 반환."""
    if not docs:
        return []
    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def translate_query(query: str) -> str:
    """한국어 쿼리를 영어로 번역해 검색 정확도를 높인다."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a cloud infrastructure expert and translator. "
            "Translate the following query to English for searching AWS Well-Architected Framework documentation. "
            "Return only the translated text, nothing else."
        )),
        HumanMessage(content=query),
    ])
    return response.content.strip()


def generate_multi_queries(requirement: str) -> list[str]:
    """사용자 요구사항에서 4개의 세분화된 검색 쿼리를 영어로 생성한다."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an AWS architect. Given a cloud infrastructure requirement, "
            "generate exactly 4 specific English search queries to find relevant AWS Well-Architected Framework guidance.\n"
            "Focus on: security, reliability, cost optimization, and performance aspects.\n"
            "Return only the 4 queries, one per line, no numbering or bullets."
        )),
        HumanMessage(content=requirement),
    ])
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return queries[:4]


# ─── 메인 검색 함수 ───────────────────────────────────────────────────────────

def search_knowledge_base(query: str, k: int = 8, top_k: int = 5) -> str:
    """Multi-query + 하이브리드 검색 + Reranker로 최고 품질 문서를 검색한다.

    Args:
        query: 사용자 쿼리 (한국어 가능)
        k: 쿼리당 1차 검색 후보 수
        top_k: Reranker 후 최종 반환 수
    """
    # 1. 원본 쿼리 번역
    translated_original = translate_query(query)
    print(f"[RAG] 원본 쿼리 번역: {translated_original}")

    # 2. Multi-query 생성 (4개)
    multi_queries = generate_multi_queries(query)
    print(f"[RAG] Multi-query {len(multi_queries)}개 생성")
    all_queries = [translated_original] + multi_queries

    # 3. 각 쿼리별 하이브리드 검색 → 후보 풀 구성 (중복 제거)
    vectorstore = load_knowledge_base()
    bm25_retriever = _get_bm25_retriever(k=k)

    seen_contents: set[str] = set()
    candidate_docs: list[Document] = []

    for q in all_queries:
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        bm25_retriever.k = k
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )
        for doc in ensemble.invoke(q):
            key = doc.page_content[:150]
            if key not in seen_contents:
                seen_contents.add(key)
                candidate_docs.append(doc)

    print(f"[RAG] 후보 총 {len(candidate_docs)}개 → Reranker로 {top_k}개 추림")

    # 4. Reranker로 원본 쿼리 기준 최종 재점수화
    reranked = rerank_documents(translated_original, candidate_docs, top_k=top_k)

    context = "\n\n---\n\n".join(doc.page_content for doc in reranked)
    return context


if __name__ == "__main__":
    """python ai_engine/rag/knowledge_base.py 로 실행하면 ChromaDB를 재빌드한다."""
    from dotenv import load_dotenv
    load_dotenv(AI_ENGINE_DIR.parent / ".env")

    vectorstore = build_knowledge_base()
    count = vectorstore._collection.count()
    print(f"\n[완료] 총 {count}개 청크 저장됨")

    if count > 0:
        print("\n[검증] 샘플 검색 테스트")
        results = vectorstore.similarity_search("VPC subnet security group best practice", k=2)
        for i, doc in enumerate(results):
            print(f"\n[청크 {i+1}] p.{doc.metadata.get('page', '?')} | pillar={doc.metadata.get('pillar', '?')} | services={doc.metadata.get('aws_services', '')}")
            print(doc.page_content[:300])
