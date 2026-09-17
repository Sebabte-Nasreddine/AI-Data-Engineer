

import os
import hashlib

import snowflake.connector
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
load_dotenv()

COLLECTION = "zomato_reviews"
BATCH_SIZE = 128


# Colonnes réelles de ZOMATO.AI.REVIEW_SEARCH_DOC (cf. review_search_doc.sql)
COLS = [
    "review_id",
    "text",              # = comment dans stg_reviews
    "city",
    "cuisine",
    "restaurant_id",
    "restaurant_name",
    "rating",
    "review_date",
    "sentiment_label",   # coalesce(..., 'unknown')
    "topic",             # coalesce(..., 'unknown')
    "key_issue",
    "content_hash",
]

qdrant = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))
bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")


# --------------------------------------------------------------------------
# Connexion Snowflake
# --------------------------------------------------------------------------

def get_connection():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "ZOMATO_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "ZOMATO"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "AI"),
    )


def fetch_batches(cursor):
    cursor.execute(f"SELECT {', '.join(COLS)} FROM ZOMATO.AI.REVIEW_SEARCH_DOC")
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break
        yield rows


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def embed_dense(texts: list[str]) -> list[list[float]]:
    # normalize_embeddings=True -> vecteurs unitaires, cohérent avec
    # Distance.COSINE côté Qdrant
    vecs = EMBED_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.tolist()

def embed_sparse(texts: list[str]):
    # retourne une liste d'objets SparseEmbedding (indices + values numpy)
    return list(bm25_model.embed(texts))


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------

def point_id(review_id) -> str:
    """UUID déterministe à partir de review_id -> upsert = pas de doublon.
    review_id peut arriver en int (NUMBER côté Snowflake) ou en str."""
    return str(hashlib.md5(str(review_id).encode()).hexdigest())

def get_existing_hashes(review_ids: list[str]) -> dict[str, str]:
    """review_id -> content_hash déjà présent dans Qdrant, pour sauter
    les reviews inchangées et ne pas repayer OpenAI pour rien."""
    result = qdrant.retrieve(
        collection_name=COLLECTION,
        ids=[point_id(rid) for rid in review_ids],
        with_payload=["review_id", "content_hash"],
    )
    return {p.payload["review_id"]: p.payload["content_hash"] for p in result}


# --------------------------------------------------------------------------
# Indexation d'un batch
# --------------------------------------------------------------------------

def index_batch(rows) -> int:
    records = [dict(zip(COLS, r)) for r in rows]

    texts = [r["text"] for r in records]
    dense_vecs = embed_dense(texts)
    sparse_vecs = embed_sparse(texts)

    points = []
    for rec, dense, sparse in zip(records, dense_vecs, sparse_vecs):
        payload = dict(rec)
        # Snowflake renvoie des objets date -> les rendre sérialisables JSON
        if payload.get("review_date") is not None:
            payload["review_date"] = payload["review_date"].isoformat()

        points.append(
            models.PointStruct(
                id=point_id(rec["review_id"]),
                vector={
                    "dense": dense,
                    "bm25": models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )

    qdrant.upsert(collection_name=COLLECTION, points=points)
    return len(points)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    conn = get_connection()
    cursor = conn.cursor()

    total_indexed = 0
    total_skipped = 0

    for batch in fetch_batches(cursor):
        review_ids = [row[0] for row in batch]
        content_hashes = [row[-1] for row in batch]  # dernière colonne = content_hash

        existing = get_existing_hashes(review_ids)

        to_index = [
            row
            for row, rid, chash in zip(batch, review_ids, content_hashes)
            if existing.get(rid) != chash
        ]
        skipped = len(batch) - len(to_index)
        total_skipped += skipped

        if to_index:
            total_indexed += index_batch(to_index)

        print(f"indexés: {total_indexed} | inchangés (sautés): {total_skipped}")

    cursor.close()
    conn.close()
    print(f"Terminé. {total_indexed} points upsertés, {total_skipped} sautés (déjà à jour).")


    

def hybrid_search(query: str, filters: dict | None = None, limit: int = 50):
    dense_vec = EMBED_MODEL.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = list(bm25_model.embed([query]))[0]

    qdrant_filter = build_filter(filters)  # -> models.Filter

    results = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", limit=limit, filter=qdrant_filter),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices.tolist(),
                                           values=sparse_vec.values.tolist()),
                using="bm25", limit=limit, filter=qdrant_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    return results.points
if __name__ == "__main__":
    main()