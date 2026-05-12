import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import db_instance, Ingest, ConfigType

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


print("deleting up test db made")
folder = Path("test_db")
if folder.exists() and folder.is_dir():
    shutil.rmtree(folder)

# try:
#     cc = Ingest()
#     if cc:
#         print("db_instance passed ✅")
# except Exception as e:
#     print(f"db_instance failed ❌: {e}")
