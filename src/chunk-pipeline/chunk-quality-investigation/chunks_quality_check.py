import json
import os
from collections import Counter

with open("data/processed/chunks_d27603b_20260529.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

parents  = [c for c in chunks if c["type"] == "parent"]
children = [c for c in chunks if c["type"] == "child"]

# Basic counts
print(f"Total chunks:   {len(chunks)}")
print(f"  Parents:      {len(parents)}")
print(f"  Children:     {len(children)}")

# Duplicate ids
ids = [c["id"] for c in chunks]
dupes = {id_: count for id_, count in Counter(ids).items() if count > 1}
print(f"Unique IDs:     {len(set(ids))}")
print(f"Duplicate IDs:  {len(dupes)}")  # must be 0

# Framework breakdown
fw = Counter(c["metadata"]["framework"] for c in children)
print(f"\nChildren by framework: {dict(fw)}")

# Child content category breakdown
cat = Counter(c["metadata"]["content_category"] for c in children)
print(f"Children by category:  {dict(cat)}")

# Text length distribution for children
print("\n── Chunk sizes ──────────────────────────────")
print("The dense & sparse embedding models we used have input limit of 512 tokens")
print("512 tokens ~ 2,000 characters. Chunks having size beyond this will face truncation")
print("leading to information loss.")
lengths = [len(c["text"]) for c in children]
print(f"Child chunk character length:")
print(f"  Min:    {min(lengths)}")
print(f"  Max:    {max(lengths)}")
print(f"  Avg:    {int(sum(lengths)/len(lengths))}")

# Detailed child chunks character-size distribution

lengths = sorted([len(c["text"]) for c in parents])

buckets = [500, 2000, 4000, 8000, 10000, 15000, 20000, 30000, 40000, float("inf")]
labels  = ["<500", "500-2k", "2k-4k", "4k-8k", "8k-10k", "10k-15k", "15k-20k", "20k-30k", "30k-40k", "40k+"]
counts  = [0] * len(labels)

for l in lengths:
    for i, b in enumerate(buckets):
        if l < b:
            counts[i] += 1
            break

total = len(children)
for label, count in zip(labels, counts):
    pct = count / total * 100
    print(f"  {label:>8}:  {count:>5} chunks  ({pct:.1f}%)")

# Chunk type wise size distribution
children = [c for c in chunks if c["type"] == "child"]

oversized = [c for c in children if len(c["text"]) > 1800]

from collections import Counter
cats = Counter(c["metadata"]["content_category"] for c in oversized)
total = len(oversized)

print(f"Oversized children (>1800 chars): {total}")
print(f"\nBy category:")
for cat, count in cats.most_common():
    pct = count / total * 100
    print(f"  {cat:>20}: {count:>4} chunks ({pct:.1f}%)")

print(f"\nBreakdown by category AND size bucket:")
buckets = [1800, 2500, 4000, 8000, 10000, 12000, 15000, float("inf")]
labels  = ["1.8k-2.5k", "2.5k-4k", "4k-8k", "8k-10k", "10k-12k", "12k-15k", "15k+"]
for cat in ["descriptive_text", "code_snippet", "structured_table"]:
    cat_chunks = [c for c in oversized if c["metadata"]["content_category"] == cat]
    if not cat_chunks:
        continue
    print(f"\n  {cat}:")
    for i, label in enumerate(labels):
        count = sum(1 for c in cat_chunks if buckets[i] <= len(c["text"]) < buckets[i+1])
        if count:
            print(f"    {label}: {count}")

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
