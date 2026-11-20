import lancedb

db = lancedb.connect("tmp/lancedb")
print("Tables found:", db.table_names())

if "Ymcp-docs-knowledge-base" in db.table_names():
    tbl = db.open_table("mcp-docs-knowledge-base")
    print("Row count:", tbl.count_rows())