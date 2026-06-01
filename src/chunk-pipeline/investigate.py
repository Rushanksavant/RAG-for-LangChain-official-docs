import json

with open("data/processed/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

children = [c for c in chunks if c["type"] == "child"]

# Sample the worst offenders from each category

def view_oversized_chunks():
    for cat in ["descriptive_text", "code_snippet"]:
        oversized = sorted(
            [c for c in children 
            if c["metadata"]["content_category"] == cat and len(c["text"]) > 1800],
            key=lambda x: len(x["text"]),
            reverse=True
        )
        print(f"\n{'='*60}")
        print(f"TOP 3 OVERSIZED: {cat}")
        print(f"{'='*60}")
        for c in oversized[:3]:
            print(f"\n  ID:        {c['id']}")
            print(f"  Length:    {len(c['text'])} chars")
            print(f"  Source:    {c['metadata'].get('source_file')}")
            print(f"  Preview:\n{c['text'][:500]}")
            print(f"  ...\n{c['text'][-200:]}")
            print(f"  ---")

# Oversized text chunks further investigation
def text_chunks():
    oversized_desc = sorted(
        [c for c in children 
        if c["metadata"]["content_category"] == "descriptive_text" and len(c["text"]) > 1800],
        key=lambda x: len(x["text"]),
        reverse=True
    )

    print(f"Oversized descriptive_text chunks: {len(oversized_desc)}")
    print()

    for c in oversized_desc[:5]:
        print(f"ID:      {c['id']}")
        print(f"Length:  {len(c['text'])} chars")
        print(f"Source:  {c['metadata'].get('source_file')}")
        print(f"Preview:\n{c['text'][:300]}")
        print(f"...")
        print(f"Mid-section:\n{c['text'][1800:2100]}")
        print(f"---")



if __name__ == "__main__":
    # view_oversized_chunks()
    text_chunks()