"""
ai/qdrant_setup.py

A executer UNE SEULE FOIS : cree la collection Qdrant avec :
  - un vecteur dense nomme "dense" (OpenAI text-embedding-3-small, 1536 dims)
  - un vecteur sparse nomme "bm25" (avec modifier=IDF, indispensable)
  - des index de payload sur les champs qu'on va filtrer

Usage:
    export QDRANT_URL=http://localhost:6333   # optionnel, valeur par defaut
    python ai/qdrant_setup.py
"""

import os
from qdrant_client import QdrantClient, models

COLLECTION = "zomato_reviews"

client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))

if client.collection_exists(COLLECTION):
    print(f"La collection '{COLLECTION}' existe déjà, rien à faire.")
else:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=512,                       # text-embedding-3-small
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,    # sans ça, BM25 degrade en simple TF
            )
        },
    )
    print(f"Collection '{COLLECTION}' créée.")

    for field, schema in [
        ("city",             models.PayloadSchemaType.KEYWORD),
        ("cuisine",          models.PayloadSchemaType.KEYWORD),
        ("restaurant_id",    models.PayloadSchemaType.KEYWORD),
        ("sentiment_label",  models.PayloadSchemaType.KEYWORD),
        ("topic",            models.PayloadSchemaType.KEYWORD),
        ("rating",           models.PayloadSchemaType.INTEGER),
        ("review_date",      models.PayloadSchemaType.DATETIME),
    ]:
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
        print(f"  index de payload créé sur '{field}'")

print("Setup terminé.")