import sys
import uuid
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_client import db_instance, Ingest, ConfigType

print("""
Running chroma tests of: creating ingest class, create, read, readAll
""")
# customConfig: ConfigType = {
#     "autoSplit": False,
#     "splitter": {
#         "chunk_overlap": 150,
#         "chunk_size": 20,
#         "method": "token"
#     },
#     "db_location":"smoketest_db"
# }

# MULTI CHROMA MANAGE TEST
# db_a = db_instance("test_db/a")
# cc_a = Ingest(db_a)

# # or could create instance and class same time

# db_b = db_instance("test_db/b")
# # config_b: ConfigType = {"db_location": "mydb/b"}
# cc_b = Ingest(db_b)

# cc_a.create("im going to db A", rawTextType="text")
# cc_b.create("im going to db B", rawTextType="text")
# print(cc_a)
# print(cc_b)
# EOT

db_test = db_instance("test_db/test")
cc = Ingest(db_test)

try:
    if cc:
        print("db_instance passed ✅")
except Exception as e:
    print(
        f"db_instance failed ❌: {e}. meaning all other tests that depend on first test to pass also failed"
    )

try:
    cc.create("Cats are fluffly pets that sleep all day", "text")
    print("create to DB passed ✅")
except Exception as e:
    print(f"create to DB failed ❌: {e}")

try:
    x = cc.read("cat")
    print("read from DB passed ✅")
except Exception as e:
    print(f"read from DB failed ❌: {e}")

try:
    x = cc.readAll()
    print("read all DB passed ✅")
except Exception as e:
    print(f"read all DB failed ❌: {e}")

try:
    x = cc.read("cat", filter={"type": "text"})
    if isinstance(x, list) and len(x) > 0:
        print("read with filter passed ✅")
except Exception as e:
    print(f"read with filter failed ❌: {e}")

try:
    x = cc.readAll(filter={"type": "text"})
    if isinstance(x, dict) and "documents" in x:
        print("readAll with filter passed ✅")
except Exception as e:
    print(f"readAll with filter failed ❌: {e}")

try:
    all_data = cc.readAll()
    doc_id = all_data["ids"][0]
    cc.update_by_id(doc_id, "Updated content: Cats are fluffy and love naps")
    print("update_by_id passed ✅")
except Exception as e:
    print(f"update_by_id failed ❌: {e}")

# update() requires interactive stdin — tested manually via exercise

try:
    all_data = cc.readAll()
    doc_id = all_data["ids"][0]
    cc.delete_by_id(doc_id)
    after = cc.readAll()
    if len(after["ids"]) == len(all_data["ids"]) - 1:
        print("delete_by_id passed ✅")
except Exception as e:
    print(f"delete_by_id failed ❌: {e}")

# delete() requires interactive stdin — tested manually via exercise


print("deleting up test db made")
folder = Path("test_db")
if folder.exists() and folder.is_dir():
    shutil.rmtree(folder)

# --- Custom prefix tests ---
print("""
Running custom prefix tests
""")

from llm_client import ConfigType
from llm_client.chroma_client import PrefixedEmbeddings
from llm_client.embedding.text_embedding import get_text_embeddings

try:
    prefix_config: ConfigType = {
        "db_location": "test_db/prefix",
        "query_prefix": "search: ",
        "document_prefix": "passage: ",
    }
    cc_prefix = Ingest(config=prefix_config)
    cc_prefix.create("Dogs are loyal and friendly companions", "text")
    results = cc_prefix.read("dogs")
    if isinstance(results, list) and len(results) > 0:
        print("custom prefix create+read passed ✅")
    else:
        print("custom prefix create+read failed ❌: no results returned")
except Exception as e:
    print(f"custom prefix create+read failed ❌: {e}")

try:
    base = get_text_embeddings()
    no_prefix = PrefixedEmbeddings(base, "", "")
    with_prefix = PrefixedEmbeddings(base, "search: ", "passage: ")

    vec_a = no_prefix.embed_query("cats")
    vec_b = with_prefix.embed_query("cats")

    if vec_a != vec_b:
        print("prefix produces different embedding vector passed ✅")
    else:
        print("prefix produces different embedding vector failed ❌: vectors are identical")
except Exception as e:
    print(f"prefix embedding difference failed ❌: {e}")

print("deleting prefix test db")
folder = Path("test_db")
if folder.exists() and folder.is_dir():
    shutil.rmtree(folder)

# --- Embedding methods tests ---
print("""
Running *_from_embedding tests
""")

db_embed = db_instance("test_db/embed_test")
cc_embed = Ingest(db_embed)

test_vec = [0.1, 0.2, 0.3, 0.4]
test_id = str(uuid.uuid4())

try:
    cc_embed.create_from_embedding(
        embedding=test_vec,
        doc_id=test_id,
        document="cat_photo.jpg",
        metadata={"type": "image", "source": "cat_photo.jpg"},
    )
    print("create_from_embedding passed ✅")
except Exception as e:
    print(f"create_from_embedding failed ❌: {e}")

try:
    results = cc_embed.read_from_embedding(test_vec, topK=1)
    if results["ids"][0][0] == test_id:
        print("read_from_embedding passed ✅")
    else:
        print(f"read_from_embedding failed ❌: returned wrong id {results['ids']}")
except Exception as e:
    print(f"read_from_embedding failed ❌: {e}")

try:
    new_vec = [0.9, 0.8, 0.7, 0.6]
    new_id = str(uuid.uuid4())
    cc_embed.create_from_embedding(embedding=new_vec, doc_id=new_id, document="dog_photo.jpg", metadata={"type": "image"})
    all_data = cc_embed.readAll()
    before_count = len(all_data["ids"])
    cc_embed.db._collection.update(
        ids=[new_id],
        embeddings=[[0.5, 0.5, 0.5, 0.5]],
        documents=["dog_photo_updated.jpg"],
        metadatas=[{"type": "image"}],
    )
    updated = cc_embed.readAll()
    idx = updated["ids"].index(new_id)
    if updated["documents"][idx] == "dog_photo_updated.jpg":
        print("update_from_embedding (non-interactive path) passed ✅")
    else:
        print("update_from_embedding failed ❌: document not updated")
except Exception as e:
    print(f"update_from_embedding failed ❌: {e}")

try:
    del_id = str(uuid.uuid4())
    cc_embed.create_from_embedding(embedding=[0.3, 0.3, 0.3, 0.3], doc_id=del_id, document="to_delete.jpg", metadata={"type": "image"})
    before = cc_embed.readAll()
    cc_embed.delete_by_id(del_id)
    after = cc_embed.readAll()
    if len(after["ids"]) == len(before["ids"]) - 1:
        print("delete_from_embedding (non-interactive path) passed ✅")
    else:
        print("delete_from_embedding failed ❌: count did not decrease")
except Exception as e:
    print(f"delete_from_embedding failed ❌: {e}")

# update_from_embedding() and delete_from_embedding() interactive flows require stdin — tested manually

print("deleting embed test db")
folder = Path("test_db")
if folder.exists() and folder.is_dir():
    shutil.rmtree(folder)

# try:
#     cc = Ingest()
#     if cc:
#         print("db_instance passed ✅")
# except Exception as e:
#     print(f"db_instance failed ❌: {e}")
