import json
import math
import os
from flask import Flask, jsonify, render_template, abort
from pathlib import Path

app = Flask(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
print(DATA_DIR)

# ─── Load data once at startup ───────────────────────────────────────────────

def load_data():
    with open(f"{DATA_DIR}/retrieval_scores_checkpoint.jsonl", encoding="utf-8") as f:
        scores = [json.loads(l) for l in f]
 
    with open(f"{DATA_DIR}/results.jsonl", encoding="utf-8") as f:
        results = [json.loads(l) for l in f]
 
    with open(f"{DATA_DIR}/knowledge_graph.json", encoding="utf-8") as f:
        kg = json.load(f)
 
    return scores, results, kg

SCORES, RESULTS, KG = load_data()

# Build a lookup: user_input → result row  (for retrieval samples)
RESULTS_BY_INPUT = {r["user_input"]: r for r in RESULTS}

# Build lookup: index → score row
SCORES_BY_INDEX = {s["index"]: s for s in SCORES}


def safe_float(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    except Exception:
        return None


def categorize_no_ret(r):
    """Bucket a no-retrieval result."""
    err = r.get("error")
    has_error = err and err not in (None, "None")
    if has_error:
        return "errored"
    ans = r.get("agent_response")
    ans_str = str(ans).strip() if ans is not None else ""
    if ans_str in ("", "None") or ans_str.startswith("Cannot"):
        return "flagged"
    return "unflagged"


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/kpis")
def kpis():
    """Mean scores for all retrieval samples."""
    def mean(vals):
        v = [safe_float(x) for x in vals]
        v = [x for x in v if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    return jsonify({
        "faithfulness": mean(s["faithfulness"] for s in SCORES),
        "context_recall": mean(s["context_recall"] for s in SCORES),
        "answer_correctness": mean(s["answer_correctness"] for s in SCORES),
        "total_samples": len(SCORES),
    })


@app.route("/api/chart_data")
def chart_data():
    """Per-sample metric scores for the line chart."""
    rows = []
    for s in SCORES:
        rows.append({
            "index": s["index"],
            "faithfulness": safe_float(s["faithfulness"]),
            "context_recall": safe_float(s["context_recall"]),
            "answer_correctness": safe_float(s["answer_correctness"]),
        })
    return jsonify(rows)


@app.route("/api/sample/<int:idx>")
def get_sample(idx):
    """Detail for a single retrieval sample by index."""
    if idx not in SCORES_BY_INDEX:
        abort(404, description=f"Index {idx} not in range 0–{len(SCORES)-1}")
    score = SCORES_BY_INDEX[idx]
    result = RESULTS_BY_INPUT.get(score["user_input"])
    if not result:
        abort(404, description="Matching result row not found for this query")

    return jsonify({
        "index": idx,
        "user_input": score["user_input"],
        "question_type": score["question_type"],
        "faithfulness": safe_float(score["faithfulness"]),
        "context_recall": safe_float(score["context_recall"]),
        "answer_correctness": safe_float(score["answer_correctness"]),
        "agent_response": result.get("agent_response"),
        "status_mssg": result.get("status_mssg"),
        "reference": result.get("reference"),
        "error": result.get("error"),
    })


@app.route("/api/no_retrieval_samples")
def no_retrieval_samples():
    """All samples where retrieval was not performed."""
    no_ret = [r for r in RESULTS if r["retrieval_performed"] is False]
    out = []
    for r in no_ret:
        out.append({
            "id": r["id"],
            "user_input": r["user_input"],
            "question_type": r["question_type"],
            "agent_response": r.get("agent_response"),
            "reference": r.get("reference"),
            "status_mssg": r.get("status_mssg"),
            "error": r.get("error"),
            "category": categorize_no_ret(r),
        })
    return jsonify(out)


@app.route("/api/knowledge_graph")
def knowledge_graph():
    """Lightweight KG summary + adjacency data for visualization."""
    nodes = KG["nodes"]
    rels = KG["relationships"]

    # Framework counts
    from collections import Counter
    fw_counts = Counter(
        n["properties"].get("document_metadata", {}).get("framework", "unknown")
        for n in nodes
    )

    # Unique entities
    entity_set = set()
    for n in nodes:
        for e in n["properties"].get("entities", []):
            entity_set.add(e)

    # Unique themes
    theme_set = set()
    for n in nodes:
        for t in n["properties"].get("themes", []):
            theme_set.add(t)

    rel_type_counts = Counter(r["type"] for r in rels)

    # Build lightweight graph for D3 (strip embeddings)
    graph_nodes = []
    for n in nodes:
        props = n["properties"]
        graph_nodes.append({
            "id": n["id"],
            "framework": props.get("document_metadata", {}).get("framework", "unknown"),
            "title": props.get("document_metadata", {}).get("title", ""),
            "section": props.get("document_metadata", {}).get("section", ""),
            "summary": props.get("summary", "")[:200],
            "entities": props.get("entities", []),
            "themes": props.get("themes", [])[:5],
        })

    graph_links = []
    for r in rels:
        if r["type"] == "summary_similarity":
            sim = r["properties"].get("summary_similarity", 0)
            # Only include stronger connections to reduce clutter
            if sim >= 0.72:
                graph_links.append({
                    "source": r["source"],
                    "target": r["target"],
                    "type": r["type"],
                    "weight": round(sim, 3),
                })

    return jsonify({
        "stats": {
            "total_nodes": len(nodes),
            "total_relationships": len(rels),
            "unique_entities": len(entity_set),
            "unique_themes": len(theme_set),
            "nodes_by_framework": dict(fw_counts),
            "relationships_by_type": dict(rel_type_counts),
        },
        "nodes": graph_nodes,
        "links": graph_links,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)