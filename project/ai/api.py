"""
ai/api.py

API FastAPI pour le pipeline RAG Zomato.

Endpoints :
    GET  /health                -> etat du service (Qdrant joignable ?)
    POST /search                -> retrieval seul (hybride + rerank), sans LLM
    POST /chat                  -> retrieval + generation (RAG complet)
    GET  /reviews/enriched      -> parcourir les reviews enrichies (pagination + filtres)

Lancer :
    cd ai
    uvicorn api:app --reload --port 8000

Doc interactive auto-generee : http://localhost:8000/docs
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from openai import OpenAI
from qdrant_client import QdrantClient, models

from hybrid_search import hybrid_search, COLLECTION
from rerank import rerank, RankedResult

app = FastAPI(title="Zomato RAG API", version="1.0")

# Ollama expose une API compatible OpenAI -> on garde le SDK openai,
# on redirige juste base_url. La clé est ignorée par Ollama mais le SDK
# exige une valeur non vide.
llm_client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
)
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

CHAT_MODEL = "qwen3:8b"
PREFETCH_LIMIT = 50
DEFAULT_TOP_K = 8


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class SearchFilters(BaseModel):
    city: Optional[str] = None
    cuisine: Optional[str] = None
    restaurant_id: Optional[int] = None
    sentiment_label: Optional[str] = None
    topic: Optional[str] = None
    rating_min: Optional[int] = None
    rating_max: Optional[int] = None
    date_from: Optional[str] = None  # "YYYY-MM-DD"
    date_to: Optional[str] = None

    def to_dict(self) -> dict:
        # seuls les champs explicitement fournis vont dans le filtre Qdrant
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SearchRequest(BaseModel):
    query: str
    filters: Optional[SearchFilters] = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)


class ReviewOut(BaseModel):
    review_id: int
    text: str
    restaurant_name: str
    city: str
    cuisine: Optional[str] = None
    rating: int
    sentiment_label: str
    topic: str
    rerank_score: float
    hybrid_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[ReviewOut]


class ChatRequest(BaseModel):
    question: str
    filters: Optional[SearchFilters] = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)


class ChatResponse(BaseModel):
    answer: str
    sources: list[ReviewOut]


class EnrichedReviewOut(BaseModel):
    review_id: int
    text: str
    restaurant_name: str
    city: str
    rating: int
    sentiment_label: str
    topic: str
    key_issue: Optional[str] = None


class EnrichedReviewsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[EnrichedReviewOut]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _to_review_out(r: RankedResult) -> ReviewOut:
    p = r.payload
    return ReviewOut(
        review_id=p["review_id"],
        text=p["text"],
        restaurant_name=p["restaurant_name"],
        city=p["city"],
        cuisine=p.get("cuisine"),
        rating=p["rating"],
        sentiment_label=p["sentiment_label"],
        topic=p["topic"],
        rerank_score=r.rerank_score,
        hybrid_score=r.score,
    )


def _retrieve(query: str, filters: Optional[SearchFilters], top_k: int) -> list[RankedResult]:
    filter_dict = filters.to_dict() if filters else None
    candidates = hybrid_search(query, filters=filter_dict, limit=PREFETCH_LIMIT)
    if not candidates:
        return []
    return rerank(query, candidates, top_k=top_k)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        info = qdrant.get_collection(COLLECTION)
        return {"status": "ok", "collection": COLLECTION, "points_count": info.points_count}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant injoignable: {e}")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    results = _retrieve(req.query, req.filters, req.top_k)
    return SearchResponse(query=req.query, results=[_to_review_out(r) for r in results])


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    top_reviews = _retrieve(req.question, req.filters, req.top_k)

    if not top_reviews:
        return ChatResponse(
            answer="Aucune review ne correspond à cette question avec ces filtres.",
            sources=[],
        )

    context = ""
    for r in top_reviews:
        p = r.payload
        context += f" ({p['city']}, {p['rating']} stars, {p['sentiment_label']}) {p['text']}\n"

    response = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "/no_think\n"
                "Answer ONLY using the customer reviews provided. "
                "Be concise. If the reviews don't cover it, say so."
            )},
            {"role": "user", "content": f"Question: {req.question}\n\nReviews:\n{context}"},
        ],
    )

    answer = response.choices[0].message.content
    # garde-fou : Qwen3 peut renvoyer un bloc <think>...</think> meme avec
    # /no_think selon la version d'Ollama -> on le retire s'il est present
    if "<think>" in answer and "</think>" in answer:
        answer = answer.split("</think>", 1)[1].strip()

    return ChatResponse(
        answer=answer,
        sources=[_to_review_out(r) for r in top_reviews],
    )


@app.get("/reviews/enriched", response_model=EnrichedReviewsPage)
def list_enriched_reviews(
    city: Optional[str] = None,
    sentiment_label: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Parcourt les reviews indexees dans Qdrant (donc deja enrichies, avec
    sentiment_label/topic a jour au moment de la derniere reindexation)."""
    must = []
    if city:
        must.append(models.FieldCondition(key="city", match=models.MatchValue(value=city)))
    if sentiment_label:
        must.append(models.FieldCondition(key="sentiment_label", match=models.MatchValue(value=sentiment_label)))
    if topic:
        must.append(models.FieldCondition(key="topic", match=models.MatchValue(value=topic)))
    qdrant_filter = models.Filter(must=must) if must else None

    total = qdrant.count(collection_name=COLLECTION, count_filter=qdrant_filter, exact=True).count

    # Qdrant scroll fonctionne par curseur (point_id), pas par offset numerique natif.
    # Pour un offset simple on scrolle en avancant par paquets -- correct pour de la
    # pagination occasionnelle, pas optimal pour scroller des dizaines de milliers
    # de pages d'affilee (utiliser plutot le curseur `next_page_offset` dans ce cas).
    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=qdrant_filter,
        limit=limit + offset,
        with_payload=True,
    )
    page = points[offset: offset + limit]

    items = [
        EnrichedReviewOut(
            review_id=p.payload["review_id"],
            text=p.payload["text"],
            restaurant_name=p.payload["restaurant_name"],
            city=p.payload["city"],
            rating=p.payload["rating"],
            sentiment_label=p.payload["sentiment_label"],
            topic=p.payload["topic"],
            key_issue=p.payload.get("key_issue"),
        )
        for p in page
    ]

    return EnrichedReviewsPage(total=total, limit=limit, offset=offset, items=items)