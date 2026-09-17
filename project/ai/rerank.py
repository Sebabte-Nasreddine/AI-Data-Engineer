from dataclasses import dataclass
from sentence_transformers import CrossEncoder
 
# ~568M params, ~1.2GB en fp16 -> largement dans les 16GB VRAM du 3080.
# Multilingue (utile si tu as des reviews en francais/arabe/anglais melanges).
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
 
_reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device="cuda")
 
 
@dataclass
class RankedResult:
    """Wrapper leger : ScoredPoint (qdrant-client) est un modele pydantic qui
    interdit l'ajout d'attributs dynamiques (rerank_score), donc on ne le
    mute pas -> on porte payload + les deux scores dans un objet a nous."""
    payload: dict
    score: float          # score hybride RRF d'origine (Qdrant)
    rerank_score: float   # score du cross-encoder
 
 
def rerank(query: str, candidates: list, top_k: int = 8) -> list[RankedResult]:
    """
    candidates : liste de points Qdrant (ceux retournes par hybrid_search()),
                 chacun avec .payload["text"] et .score.
 
    Retourne les `top_k` RankedResult les mieux classes par le cross-encoder.
    """
    if not candidates:
        return []
 
    pairs = [(query, c.payload["text"]) for c in candidates]
    scores = _reranker.predict(pairs)  # -> array numpy de floats
 
    ranked_results = [
        RankedResult(payload=c.payload, score=c.score, rerank_score=float(s))
        for c, s in zip(candidates, scores)
    ]
 
    ranked_results.sort(key=lambda r: r.rerank_score, reverse=True)
    return ranked_results[:top_k]
 
 
if __name__ == "__main__":
    from hybrid_search import hybrid_search
 
    query = "livraison en retard et plats froids"
    candidates = hybrid_search(query, filters=None, limit=50)
    print(f"{len(candidates)} candidats hybrides récupérés")
 
    top = rerank(query, candidates, top_k=8)
    for c in top:
        print(f"[rerank={c.rerank_score:.4f} | hybride={c.score:.4f}] "
              f"{c.payload['restaurant_name']} — {c.payload['text'][:80]}")