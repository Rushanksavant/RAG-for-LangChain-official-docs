import json
import os
from collections import Counter

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

parents  = [c for c in chunks if c["type"] == "parent"]
children = [c for c in chunks if c["type"] == "child"]

# Basic counts
print(f"Total chunks:   {len(chunks)}")
print(f"  Parents:      {len(parents)}")
print(f"  Children:     {len(children)}")

# Framework breakdown
fw = Counter(c["metadata"]["framework"] for c in children)
print(f"\nChildren by framework: {dict(fw)}")

# Child content category breakdown
cat = Counter(c["metadata"]["content_category"] for c in children)
print(f"Children by category:  {dict(cat)}")

# Text length distribution for children
lengths = [len(c["text"]) for c in children]
print(f"\nChild text length:")
print(f"  Min:    {min(lengths)}")
print(f"  Max:    {max(lengths)}")
print(f"  Avg:    {int(sum(lengths)/len(lengths))}")

# Sample: 2 parents + their children
print("\n── Sample parent ──────────────────────────────")
p = parents[10]
print(f"ID:   {p['id']}")
print(f"Text preview:\n{p['text'][:300]}")
kids = [c for c in children if c["metadata"]["parent_id"] == p["id"]]
print(f"Children: {len(kids)}")
for k in kids[:3]:
    print(f"  [{k['metadata']['content_category']}] {k['text'][:120]!r}")

# Check: any child with missing parent_id?
orphans = [c for c in children if not c["metadata"].get("parent_id")]
print(f"\nOrphan children (missing parent_id): {len(orphans)}")

# Check: any very short children (potential noise)
tiny = [c for c in children if len(c["text"]) < 50]
print(f"Tiny children (<50 chars):           {len(tiny)}")
if tiny:
    for t in tiny[:5]:
        print(f"  {t['text']!r}")
