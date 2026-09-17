"""
ai/hybrid_search.py

Recherche hybride sur la collection Qdrant "zomato_reviews" :
  - vecteur dense  (SentenceTransformer distiluse-base-multilingual-cased-v2) -> semantique
  - vecteur sparse (BM25 via fastembed)                                       -> lexical
  - fusion server-side par Qdrant (RRF)
  - filtre de metadonnees applique AVANT la fusion (sur chaque prefetch),
    pas apres, pour ne pas fausser le classement des survivants.

Usage:
    from ai.hybrid_search import hybrid_search

    results = hybrid_search(
        "livraison en retard et plats froids",
        filters={"city": "Casablanca", "sentiment_label": "negative", "rating_max": 2},
        limit=50,
    )
"""

import os
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

COLLECTION = "zomato_reviews"

EMBED_MODEL = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


# --------------------------------------------------------------------------
# Construction du filtre Qdrant
# --------------------------------------------------------------------------

# Champs à égalité exacte (payload indexé en KEYWORD dans qdrant_setup.py)
_KEYWORD_FIELDS = {"city", "cuisine", "restaurant_id", "sentiment_label", "topic"}


def build_filter(filters: dict | None) -> models.Filter | None:
    """
    Traduit un dict simple en models.Filter Qdrant.

    Clés acceptées :
      - city, cuisine, restaurant_id, sentiment_label, topic
            -> egalite exacte. Accepte une valeur unique ou une liste
               (liste => OR implicite via `should` interne a MatchAny).
      - rating_min, rating_max
            -> range sur le champ payload `rating`.
      - date_from, date_to
            -> range sur le champ payload `review_date` (format ISO "YYYY-MM-DD").

    Retourne None si `filters` est vide/None -> pas de filtre, comportement
    normal (recherche sur toute la collection).
    """
    if not filters:
        return None

    must: list[models.Condition] = []

    for field in _KEYWORD_FIELDS:
        if field not in filters:
            continue
        value = filters[field]
        if isinstance(value, (list, tuple, set)):
            must.append(
                models.FieldCondition(key=field, match=models.MatchAny(any=list(value)))
            )
        else:
            must.append(
                models.FieldCondition(key=field, match=models.MatchValue(value=value))
            )

    if "rating_min" in filters or "rating_max" in filters:
        must.append(
            models.FieldCondition(
                key="rating",
                range=models.Range(
                    gte=filters.get("rating_min"),
                    lte=filters.get("rating_max"),
                ),
            )
        )

    if "date_from" in filters or "date_to" in filters:
        must.append(
            models.FieldCondition(
                key="review_date",
                range=models.DatetimeRange(
                    gte=filters.get("date_from"),
                    lte=filters.get("date_to"),
                ),
            )
        )

    if not must:
        return None

    return models.Filter(must=must)


# --------------------------------------------------------------------------
# Recherche hybride
# --------------------------------------------------------------------------

def hybrid_search(query: str, filters: dict | None = None, limit: int = 50):
    """
    Retourne les `limit` points les plus pertinents pour `query`, en fusionnant
    recherche dense et sparse par RRF, sous contrainte du filtre `filters`.
    """
    dense_vec = EMBED_MODEL.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = list(bm25_model.embed([query]))[0]

    qdrant_filter = build_filter(filters)

    results = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=limit,
                filter=qdrant_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="bm25",
                limit=limit,
                filter=qdrant_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    return results.points


if __name__ == "__main__":
    # test rapide en ligne de commande
    points = hybrid_search(
        "Gravy spilled all over the bag",
        filters={"sentiment_label": "negative"},
        limit=10,
    )
    for p in points:
        print(f"[{p.score:.4f}] {p.payload['restaurant_name']} — {p.payload['text'][:80]}")