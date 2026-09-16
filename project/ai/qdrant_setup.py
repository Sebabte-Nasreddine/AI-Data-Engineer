from qdrant_client import QdrantClient, models

COLLECTION = "zomato_reviews"
client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name=COLLECTION,
    vectors_config={
        "dense": models.VectorParams(
            size=512,                          # text-embedding-3-small
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF,       # <- see note below
        )
    },
)

# Payload indexes: required for pre-filtering to be fast.
for field, schema in [
    ("city",          models.PayloadSchemaType.KEYWORD),
    ("restaurant_id", models.PayloadSchemaType.KEYWORD),
    ("cuisine",       models.PayloadSchemaType.KEYWORD),
    ("sentiment",     models.PayloadSchemaType.KEYWORD),
    ("topic",         models.PayloadSchemaType.KEYWORD),
    ("rating",        models.PayloadSchemaType.INTEGER),
    ("review_date",   models.PayloadSchemaType.DATETIME),
]:
    client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)