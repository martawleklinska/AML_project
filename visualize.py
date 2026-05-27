"""
visualize.py
------------
Parse pan12_dataset/{split}.json, extract rule/risk metadata,
write sorted JSON + readable TXT + matplotlib histograms.

Outputs (all in OUTPUT_DIR):
  analysis_{split}.json        -- all conversations with metadata, sorted by risk score desc
  debug_{split}_{suffix}.txt   -- human-readable conversations
  hist_risk_level.png          -- risk level distribution by label
  hist_top_categories.png      -- top 20 most triggered rule categories
  hist_category_count.png      -- unique categories per conversation
  hist_severity_dist.png       -- severity-3 hit count per conversation
  hist_escalation_arc.png      -- escalation / arc flag rates by label

CONFIG at bottom of file.
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_DIR = "pan12_dataset"
SPLIT       = "train"   # "train" / "val" / "test"
MAX_CONVS   = None      # None = all
ONLY_POS    = False     # False = include both grooming and benign
OUTPUT_DIR  = "debug"
# ---------------------------------------------------------------------------

# Regex patterns for prefix tokens
_RE_RISK  = re.compile(r"\[RISK:(HIGH|MEDIUM|LOW)\]")
_RE_CATS  = re.compile(r"\[CATS:([^\]]*)\]")
_RE_ESC   = re.compile(r"\[ESC:([01])\]")
_RE_ARC   = re.compile(r"\[ARC:([01])\]")
_RE_DOM   = re.compile(r"\[DOM:([01])\]")
_RE_RULE  = re.compile(r"\[RULE:([a-z_]+):([123])\]")
_RE_SEP   = re.compile(r" \[SEP\] ")

_RISK_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_record(record: dict) -> dict:
    """
    Extract metadata from the flat text string.
    Returns enriched record dict.
    """
    text  = record["text"]
    label = record["label"]

    # --- Prefix tokens ---
    risk_m  = _RE_RISK.search(text)
    cats_m  = _RE_CATS.search(text)
    esc_m   = _RE_ESC.search(text)
    arc_m   = _RE_ARC.search(text)
    dom_m   = _RE_DOM.search(text)

    risk_level = risk_m.group(1)  if risk_m  else "UNKNOWN"
    cats_str   = cats_m.group(1)  if cats_m  else "none"
    esc_flag   = esc_m.group(1) == "1" if esc_m else False
    arc_flag   = arc_m.group(1) == "1" if arc_m else False
    dom_flag   = dom_m.group(1) == "1" if dom_m else False

    categories = [c for c in cats_str.split(",") if c and c != "none"]

    # --- Per-message rule hits ---
    rule_hits: list[dict] = []          # {category, severity, sev_int}
    cat_counts: Counter   = Counter()
    sev3_count = 0

    # Strip prefix, then split on [SEP]
    body = _RE_RISK.sub("", text)
    body = _RE_CATS.sub("", body)
    body = _RE_ESC.sub("", body)
    body = _RE_ARC.sub("", body)
    body = _RE_DOM.sub("", body).strip()

    messages = []
    for part in _RE_SEP.split(body):
        part = part.strip()
        if not part:
            continue

        # Extract rule tags from this message part
        rules_in_msg = []
        for m in _RE_RULE.finditer(part):
            cat   = m.group(1)
            sev   = int(m.group(2))
            rules_in_msg.append({"category": cat, "severity": sev})
            cat_counts[cat] += 1
            if sev == 3:
                sev3_count += 1
            rule_hits.append({"category": cat, "severity": sev})

        # Clean rule tags from display text
        clean = _RE_RULE.sub("", part).strip()

        role = "PRED" if clean.startswith("[PRED]") else "USER"
        msg_text = re.sub(r"^\[(PRED|USER)\]\s*", "", clean).strip()

        messages.append({
            "role":  role,
            "text":  msg_text,
            "rules": rules_in_msg,
        })

    return {
        "conversation_id": record["conversation_id"],
        "label":           label,
        "label_str":       "GROOMING" if label == 1 else "benign",
        "risk_level":      risk_level,
        "risk_order":      _RISK_ORDER.get(risk_level, 0),
        "escalation":      esc_flag,
        "arc":             arc_flag,
        "dominance":       dom_flag,
        "categories":      categories,
        "n_categories":    len(categories),
        "cat_counts":      dict(cat_counts),
        "total_rule_hits": sum(cat_counts.values()),
        "sev3_hits":       sev3_count,
        "n_messages":      len(messages),
        "messages":        messages,
    }


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

def format_conversation(rec: dict, idx: int) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(
        f"#{idx}  [{rec['label_str']}]  RISK:{rec['risk_level']}"
        + ("  ESC" if rec["escalation"] else "")
        + ("  ARC" if rec["arc"] else "")
        + ("  DOM" if rec["dominance"] else "")
    )
    lines.append(f"id={rec['conversation_id']}")
    if rec["categories"]:
        lines.append(f"cats={', '.join(rec['categories'])}")
    if rec["cat_counts"]:
        top = sorted(rec["cat_counts"].items(), key=lambda x: -x[1])
        lines.append("rules=" + "  ".join(f"{c}:{n}" for c, n in top))
    lines.append("-" * 70)

    for msg in rec["messages"]:
        role   = "[PRED]" if msg["role"] == "PRED" else "[USER]"
        rules  = ""
        if msg["rules"]:
            tags  = " ".join(f"[{r['category']}:{r['severity']}]" for r in msg["rules"])
            rules = f"  {tags}"
        lines.append(f"{role}{rules}  {msg['text']}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_risk_level(records: list, out: Path) -> None:
    """Risk level distribution split by grooming vs benign."""
    levels = ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
    pos_counts = Counter(r["risk_level"] for r in records if r["label"] == 1)
    neg_counts = Counter(r["risk_level"] for r in records if r["label"] == 0)

    x  = np.arange(len(levels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, [pos_counts.get(l, 0) for l in levels], w, label="grooming", color="#e74c3c")
    ax.bar(x + w/2, [neg_counts.get(l, 0) for l in levels], w, label="benign",   color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Conversations")
    ax.set_title("Risk level distribution by label")
    ax.legend()
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, out / "hist_risk_level.png")


def plot_top_categories(records: list, out: Path, top_n: int = 20) -> None:
    """Top N most triggered rule categories across all grooming conversations."""
    total: Counter = Counter()
    for r in records:
        if r["label"] == 1:
            total.update(r["cat_counts"])

    if not total:
        return

    labels_, values = zip(*total.most_common(top_n))
    y = np.arange(len(labels_))

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels_) * 0.4)))
    ax.barh(y, values, color="#e67e22")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total hits (grooming conversations)")
    ax.set_title(f"Top {top_n} rule categories in grooming conversations")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, out / "hist_top_categories.png")


def plot_category_count(records: list, out: Path) -> None:
    """Distribution of unique category count per conversation."""
    pos_counts = [r["n_categories"] for r in records if r["label"] == 1]
    neg_counts = [r["n_categories"] for r in records if r["label"] == 0]

    max_val = max(max(pos_counts, default=0), max(neg_counts, default=0))
    bins    = range(0, max_val + 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pos_counts, bins=bins, alpha=0.6, label="grooming", color="#e74c3c", align="left")
    ax.hist(neg_counts, bins=bins, alpha=0.6, label="benign",   color="#3498db", align="left")
    ax.set_xlabel("Unique categories triggered")
    ax.set_ylabel("Conversations")
    ax.set_title("Unique rule categories per conversation")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, out / "hist_category_count.png")


def plot_severity3_dist(records: list, out: Path) -> None:
    """Distribution of severity-3 hits per conversation."""
    pos_sev3 = [r["sev3_hits"] for r in records if r["label"] == 1]
    neg_sev3 = [r["sev3_hits"] for r in records if r["label"] == 0]

    max_val = max(max(pos_sev3, default=0), max(neg_sev3, default=0), 1)
    bins    = range(0, min(max_val + 2, 20))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pos_sev3, bins=bins, alpha=0.6, label="grooming", color="#bc1300", align="left")
    ax.hist(neg_sev3, bins=bins, alpha=0.6, label="benign",   color="#82b5d7", align="left")
    ax.set_xlabel("Severity-3 rule hits")
    ax.set_ylabel("Conversations")
    ax.set_title("Severity-3 hits per conversation")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, out / "hist_severity3.png")


def plot_escalation_arc(records: list, out: Path) -> None:
    """Escalation and arc flag rates by label."""
    def rate(recs, flag):
        if not recs:
            return 0.0
        return sum(1 for r in recs if r[flag]) / len(recs) * 100

    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    flags  = ["escalation", "arc", "dominance"]
    labels_ = ["Escalation", "Stage Arc", "Dominance"]
    pos_rates = [rate(pos, f) for f in flags]
    neg_rates = [rate(neg, f) for f in flags]

    x = np.arange(len(flags))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w/2, pos_rates, w, label="grooming", color="#e74c3c")
    ax.bar(x + w/2, neg_rates, w, label="benign",   color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_)
    ax.set_ylabel("% of conversations")
    ax.set_title("Structural signal rates by label")
    ax.legend()
    ax.set_ylim(0, 100)
    _save(fig, out / "hist_escalation_arc.png")


def plot_total_hits_dist(records: list, out: Path) -> None:
    """Total rule hits per conversation (log-scale friendly)."""
    pos_hits = [r["total_rule_hits"] for r in records if r["label"] == 1]
    neg_hits = [r["total_rule_hits"] for r in records if r["label"] == 0]

    max_val = max(max(pos_hits, default=0), max(neg_hits, default=0), 1)
    bins    = np.linspace(0, max_val + 1, min(40, max_val + 2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pos_hits, bins=bins, alpha=0.6, label="grooming", color="#e74c3c")
    ax.hist(neg_hits, bins=bins, alpha=0.6, label="benign",   color="#3498db")
    ax.set_xlabel("Total rule hits per conversation")
    ax.set_ylabel("Conversations")
    ax.set_title("Total rule hits distribution")
    ax.legend()
    _save(fig, out / "hist_total_hits.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    src = Path(DATASET_DIR) / f"{SPLIT}.json"
    with open(src, encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} conversations from {src}")

    # Parse all records
    records = [parse_record(r) for r in raw]

    # Filter
    if ONLY_POS:
        records = [r for r in records if r["label"] == 1]
    if MAX_CONVS is not None:
        records = records[:MAX_CONVS]

    # Sort by risk_order desc, then sev3_hits desc, then total_rule_hits desc
    records.sort(key=lambda r: (r["risk_order"], r["sev3_hits"], r["total_rule_hits"]), reverse=True)

    out = Path(OUTPUT_DIR)
    out.mkdir(exist_ok=True)

    # --- JSON output ---
    json_path = out / f"analysis_{SPLIT}.json"
    # Drop messages from JSON to keep it readable (they're in the TXT)
    slim = [{k: v for k, v in r.items() if k != "messages"} for r in records]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"  saved {json_path.name}  ({len(slim)} records)")

    # --- TXT output ---
    suffix   = "pos" if ONLY_POS else "all"
    txt_path = out / f"debug_{SPLIT}_{suffix}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records, 1):
            f.write(format_conversation(rec, idx))
    print(f"  saved {txt_path.name}")

    # --- Histograms (always over full split, not filtered) ---
    print("Generating histograms...")
    all_parsed = [parse_record(r) for r in raw]
    plot_risk_level(all_parsed,       out)
    plot_top_categories(all_parsed,   out)
    plot_category_count(all_parsed,   out)
    plot_severity3_dist(all_parsed,   out)
    plot_escalation_arc(all_parsed,   out)
    plot_total_hits_dist(all_parsed,  out)

    # --- Console summary ---
    pos = [r for r in all_parsed if r["label"] == 1]
    neg = [r for r in all_parsed if r["label"] == 0]
    print(f"\nSummary ({SPLIT} split):")
    print(f"  grooming : {len(pos):6d}  |  HIGH={sum(1 for r in pos if r['risk_level']=='HIGH'):5d}  "
          f"MEDIUM={sum(1 for r in pos if r['risk_level']=='MEDIUM'):5d}  "
          f"LOW={sum(1 for r in pos if r['risk_level']=='LOW'):5d}")
    print(f"  benign   : {len(neg):6d}  |  HIGH={sum(1 for r in neg if r['risk_level']=='HIGH'):5d}  "
          f"MEDIUM={sum(1 for r in neg if r['risk_level']=='MEDIUM'):5d}  "
          f"LOW={sum(1 for r in neg if r['risk_level']=='LOW'):5d}")

    all_cat: Counter = Counter()
    for r in pos:
        all_cat.update(r["cat_counts"])
    print(f"\n  Top 5 categories in grooming conversations:")
    for cat, cnt in all_cat.most_common(5):
        print(f"    {cat:<30s} {cnt:6d} hits")


if __name__ == "__main__":
    main()
