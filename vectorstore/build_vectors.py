import json
import sqlite3
import uuid
from qdrant_client.models import PayloadSchemaType
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from config.settings import DB_PATH, EMBEDDING_MODEL, EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_URL, QDRANT_API_KEY

#Converts a JSON string into a Python list
def as_list(value):
    try:
        return [str(v).strip() for v in json.loads(value or "[]") if str(v).strip()]
    except json.JSONDecodeError:
        return []

#converts JSON strings into Python objects
def as_json(value, default):
    try:
        return json.loads(value) if value else default
    except json.JSONDecodeError:
        return default

#Combines the important fields into one text string
def build_text(r):
    parts = [f"name: {r['name']},"]
    if r["primary_category"]:
        parts.append(f"Category: {r['primary_category']}.")
    if as_list(r["tags"]):
        parts.append(f"Tags: {', '.join(as_list(r['tags']))}.")
    if as_list(r["ingredients"]):
        parts.append(f"Ingredients: {', '.join(as_list(r['ingredients']))}.")
    if r["description"]:
        parts.append(f"Description: {r['description']}")
    return " ".join(parts)


# Read items from the database and Converts each database row into a Python dictionary
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
items = [dict(r) for r in conn.execute("SELECT * FROM items WHERE active = 1")]
conn.close()

# loads the embedding model one (string per item -> one vector per item)
model = SentenceTransformer(
    EMBEDDING_MODEL,
    config_kwargs={"vision_config": None, "audio_config": None},
    model_kwargs={"torch_dtype": torch.float32},
)

#Convert menu items into vectors
vectors = model.encode(
    [build_text(i) for i in items],
    prompt_name="Document",
    truncate_dim=EMBEDDING_DIM,
    normalize_embeddings=True,
    batch_size=16,
)

#Connect to Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

#Create a collection if it doesn't exist
if not client.collection_exists(QDRANT_COLLECTION):
    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

#Create payload indexes
for field, schema in [
    ("primary_category", PayloadSchemaType.KEYWORD),
    ("ingredients", PayloadSchemaType.KEYWORD),
    ("price", PayloadSchemaType.FLOAT),
    ("calories", PayloadSchemaType.INTEGER),
    ("popular", PayloadSchemaType.BOOL),
]:
    client.create_payload_index(QDRANT_COLLECTION, field_name=field, field_schema=schema)
#Prepare the points
points = [
    PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, i["id"])),
        vector=v.tolist(),
        payload={
    "name": i["name"],
    "primary_category": i["primary_category"],
    "ingredients": as_list(i["ingredients"]),
    "price": i["price"],
    "calories": i["calories"],
    "popular": bool(i["popular"]),
    "price_imputed": bool(i["price_imputed"]),
    "calories_imputed": bool(i["calories_imputed"]),
    "modifier_groups": as_json(i["modifier_groups"], []),
    "text": build_text(i),  # the exact string that was embedded
},
    )
    for i, v in zip(items, vectors)
]

# Upload the points in batches
BATCH = 16
for start in range(0, len(points), BATCH):
    client.upsert(QDRANT_COLLECTION, points=points[start:start + BATCH])
    print(f"  uploaded {min(start + BATCH, len(points))}/{len(points)}")

client.close()
print(f"Stored {len(items)} vectors in Qdrant collection {QDRANT_COLLECTION}")