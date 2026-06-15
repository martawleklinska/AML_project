"""
Plotnine visualizer for pan12_dataset JSON files.

Input:
  pan12_dataset/train.json
  pan12_dataset/val.json
  pan12_dataset/test.json

Output:
  debug_p9/analysis_{split}.json
  debug_p9/debug_{split}_{suffix}.txt
  debug_p9/plots/*.png
  debug_p9/plots/*.pdf
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotnine as p9

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_DIR = "pan12_dataset"
SPLITS = ["train", "val", "test"]
MAX_CONVS = None
ONLY_POS = False
OUTPUT_DIR = "debug_p9"
PLOT_DIR = "plots"
TOP_N_CATEGORIES = 20
SAVE_PNG = False
SAVE_PDF = True
# ---------------------------------------------------------------------------

# Thesis palette
CL_CHARCOAL = "#242424"
CL_ASH = "#d8d8d8"
CL_SLATE = "#5f6b7a"
CL_BLUE = "#3b6ea8"
CL_BLUE_LIGHT = "#9bbbd8"
CL_RED = "#b23a3a"
CL_RED_LIGHT = "#df9a92"
CL_AMBER = "#c77c2e"
CL_GREEN = "#5d8a66"
CL_PURPLE = "#7768ae"
CL_GRAY = "#8a8a8a"

LABEL_COLORS = {
    "grooming": CL_RED,
    "benign": CL_BLUE,
}

RISK_COLORS = {
    "HIGH": CL_RED,
    "MEDIUM": CL_AMBER,
    "LOW": CL_GREEN,
    "UNKNOWN": CL_GRAY,
}

SIGNAL_COLORS = {
    "Escalation": CL_RED,
    "Stage arc": CL_AMBER,
    "Dominance": CL_BLUE,
}


def theme_thesis(width=5.9, height=3.6):
    return (
        p9.theme_minimal(base_size=10, base_family="serif")
        + p9.theme(
            figure_size=(width, height),
            dpi=300,
            axis_title=p9.element_text(size=10, color=CL_CHARCOAL),
            axis_text=p9.element_text(size=8, color=CL_CHARCOAL),
            legend_title=p9.element_text(size=9, color=CL_CHARCOAL),
            legend_text=p9.element_text(size=8, color=CL_CHARCOAL),
            strip_text=p9.element_text(size=8, weight="bold", color=CL_CHARCOAL),
            panel_grid_major=p9.element_line(size=0.25, color=CL_ASH),
            panel_grid_minor=p9.element_blank(),
            panel_background=p9.element_rect(fill="white", color=None),
            plot_background=p9.element_rect(fill="white", color=None),
            plot_title=p9.element_text(size=11, weight="bold", color=CL_CHARCOAL),
            plot_subtitle=p9.element_text(size=9, color=CL_SLATE),
            legend_position="right",
        )
    )

# Regex patterns for metadata tokens
_RE_RISK = re.compile(r"\[RISK:(HIGH|MEDIUM|LOW)\]")
_RE_CATS = re.compile(r"\[CATS:([^\]]*)\]")
_RE_ESC = re.compile(r"\[ESC:([01])\]")
_RE_ARC = re.compile(r"\[ARC:([01])\]")
_RE_DOM = re.compile(r"\[DOM:([01])\]")
_RE_RULE = re.compile(r"\[RULE:([a-z_]+):([123])(?::([a-z_]+))?\]")
_RE_SEP = re.compile(r" \[SEP\] ")
_RE_PREFIX_TAGS = [
    _RE_RISK,
    _RE_CATS,
    _RE_ESC,
    _RE_ARC,
    _RE_DOM,
]

RISK_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
RISK_LEVELS = ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
LABEL_LEVELS = ["grooming", "benign"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_record(record: dict, split: str) -> dict:
    text = str(record.get("text", ""))
    label = int(record.get("label", 0))

    risk_m = _RE_RISK.search(text)
    cats_m = _RE_CATS.search(text)
    esc_m = _RE_ESC.search(text)
    arc_m = _RE_ARC.search(text)
    dom_m = _RE_DOM.search(text)

    risk_level = risk_m.group(1) if risk_m else "UNKNOWN"
    cats_str = cats_m.group(1) if cats_m else "none"
    categories = [c for c in cats_str.split(",") if c and c != "none"]

    body = text
    for pat in _RE_PREFIX_TAGS:
        body = pat.sub("", body)
    body = body.strip()

    messages = []
    cat_counts = Counter()
    rule_method_counts = Counter()
    sev_counts = Counter()

    for part in _RE_SEP.split(body):
        part = part.strip()
        if not part:
            continue

        rules_in_msg = []
        for m in _RE_RULE.finditer(part):
            cat = m.group(1)
            sev = int(m.group(2))
            method = m.group(3) or "regex"
            rules_in_msg.append({
                "category": cat,
                "severity": sev,
                "method": method,
            })
            cat_counts[cat] += 1
            rule_method_counts[method] += 1
            sev_counts[sev] += 1

        clean = _RE_RULE.sub("", part).strip()
        role = "PRED" if clean.startswith("[PRED]") else "USER"
        msg_text = re.sub(r"^\[(PRED|USER)\]\s*", "", clean).strip()
        messages.append({"role": role, "text": msg_text, "rules": rules_in_msg})

    return {
        "split": split,
        "conversation_id": record.get("conversation_id", ""),
        "label": label,
        "label_str": "grooming" if label == 1 else "benign",
        "risk_level": risk_level,
        "risk_order": RISK_ORDER.get(risk_level, 0),
        "escalation": bool(esc_m and esc_m.group(1) == "1"),
        "arc": bool(arc_m and arc_m.group(1) == "1"),
        "dominance": bool(dom_m and dom_m.group(1) == "1"),
        "categories": categories,
        "n_categories": len(categories),
        "cat_counts": dict(cat_counts),
        "rule_method_counts": dict(rule_method_counts),
        "sev1_hits": int(sev_counts[1]),
        "sev2_hits": int(sev_counts[2]),
        "sev3_hits": int(sev_counts[3]),
        "total_rule_hits": int(sum(cat_counts.values())),
        "n_messages": len(messages),
        "n_pred_messages": sum(1 for m in messages if m["role"] == "PRED"),
        "n_user_messages": sum(1 for m in messages if m["role"] == "USER"),
        "messages": messages,
    }


def load_split(split: str) -> list[dict]:
    path = Path(DATASET_DIR) / f"{split}.json"
    if not path.exists():
        print(f"skip missing split: {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}")
    records = [parse_record(r, split) for r in raw]
    print(f"loaded {split}: {len(records)} records")
    return records


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def records_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "split": r["split"],
            "conversation_id": r["conversation_id"],
            "label": r["label"],
            "label_str": r["label_str"],
            "risk_level": r["risk_level"],
            "risk_order": r["risk_order"],
            "escalation": r["escalation"],
            "arc": r["arc"],
            "dominance": r["dominance"],
            "n_categories": r["n_categories"],
            "total_rule_hits": r["total_rule_hits"],
            "sev1_hits": r["sev1_hits"],
            "sev2_hits": r["sev2_hits"],
            "sev3_hits": r["sev3_hits"],
            "n_messages": r["n_messages"],
            "n_pred_messages": r["n_pred_messages"],
            "n_user_messages": r["n_user_messages"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["risk_level"] = pd.Categorical(df["risk_level"], RISK_LEVELS, ordered=True)
    df["label_str"] = pd.Categorical(df["label_str"], LABEL_LEVELS, ordered=True)
    return df


def category_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for cat, count in r["cat_counts"].items():
            rows.append({
                "split": r["split"],
                "conversation_id": r["conversation_id"],
                "label": r["label"],
                "label_str": r["label_str"],
                "category": cat,
                "count": count,
            })
    return pd.DataFrame(rows)


def method_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for method, count in r["rule_method_counts"].items():
            rows.append({
                "split": r["split"],
                "conversation_id": r["conversation_id"],
                "label_str": r["label_str"],
                "method": method,
                "count": count,
            })
    return pd.DataFrame(rows)


def signal_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    flags = [
        ("Escalation", "escalation"),
        ("Stage arc", "arc"),
        ("Dominance", "dominance"),
    ]
    for split in sorted(set(r["split"] for r in records)):
        for label in LABEL_LEVELS:
            subset = [r for r in records if r["split"] == split and r["label_str"] == label]
            n = len(subset)
            for signal_name, key in flags:
                value = 100.0 * sum(1 for r in subset if r[key]) / n if n else 0.0
                rows.append({
                    "split": split,
                    "label_str": label,
                    "signal": signal_name,
                    "rate": value,
                    "n": n,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["label_str"] = pd.Categorical(df["label_str"], LABEL_LEVELS, ordered=True)
        df["signal"] = pd.Categorical(df["signal"], ["Escalation", "Stage arc", "Dominance"], ordered=True)
    return df


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_conversation(rec: dict, idx: int) -> str:
    lines = []
    lines.append("=" * 78)
    header = f"#{idx} [{rec['split']}] [{rec['label_str']}] RISK:{rec['risk_level']}"
    if rec["escalation"]:
        header += " ESC"
    if rec["arc"]:
        header += " ARC"
    if rec["dominance"]:
        header += " DOM"
    lines.append(header)
    lines.append(f"id={rec['conversation_id']}")
    if rec["categories"]:
        lines.append("cats=" + ", ".join(rec["categories"]))
    if rec["cat_counts"]:
        top = sorted(rec["cat_counts"].items(), key=lambda x: (-x[1], x[0]))
        lines.append("rules=" + "  ".join(f"{cat}:{cnt}" for cat, cnt in top))
    lines.append("-" * 78)

    for msg in rec["messages"]:
        role = "[PRED]" if msg["role"] == "PRED" else "[USER]"
        tags = ""
        if msg["rules"]:
            tags = " " + " ".join(
                f"[{r['category']}:{r['severity']}:{r['method']}]" for r in msg["rules"]
            )
        lines.append(f"{role}{tags} {msg['text']}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(records: list[dict], split: str, out: Path) -> None:
    split_records = [r for r in records if r["split"] == split]
    if ONLY_POS:
        split_records = [r for r in split_records if r["label"] == 1]
    if MAX_CONVS is not None:
        split_records = split_records[:MAX_CONVS]
    split_records.sort(
        key=lambda r: (r["risk_order"], r["sev3_hits"], r["total_rule_hits"], r["n_categories"]),
        reverse=True,
    )

    slim = [{k: v for k, v in r.items() if k != "messages"} for r in split_records]
    json_path = out / f"analysis_{split}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)

    suffix = "pos" if ONLY_POS else "all"
    txt_path = out / f"debug_{split}_{suffix}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(split_records, 1):
            f.write(format_conversation(rec, idx))

    print(f"saved {json_path}")
    print(f"saved {txt_path}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def save_plot(plot, path_base: Path, width=8, height=5) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        png = path_base.with_suffix(".png")
        plot.save(png, width=width, height=height, dpi=300, verbose=False)
        print(f"saved {png}")
    if SAVE_PDF:
        pdf = path_base.with_suffix(".pdf")
        plot.save(pdf, width=width, height=height, dpi=300, verbose=False)
        print(f"saved {pdf}")


def pct_labels(df: pd.DataFrame, group_cols: list[str], count_col="n") -> pd.DataFrame:
    out = df.copy()
    totals = out.groupby(group_cols, observed=False)[count_col].transform("sum").astype(float)
    counts = pd.to_numeric(out[count_col], errors="coerce").fillna(0.0).astype(float)
    denom = totals.where(totals != 0)
    out["pct"] = (counts / denom * 100.0).fillna(0.0).astype(float)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_risk_distribution(df: pd.DataFrame, plot_dir: Path) -> None:
    d = df.groupby(["split", "label_str", "risk_level"], observed=False).size().reset_index(name="n")
    d = pct_labels(d, ["split", "label_str"])

    g = (
        p9.ggplot(d, p9.aes("risk_level", "n", fill="label_str"))
        + p9.geom_col(position=p9.position_dodge(width=0.75), width=0.65)
        + p9.facet_wrap("~split", scales="free_y")
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.labs(
            title="Risk level distribution",
            subtitle="Counts by split and ground-truth label",
            x="Risk level",
            y="Conversations",
            fill="Label",
        )
        + theme_thesis(8, 6)
    )
    save_plot(g, plot_dir / "risk_level_distribution", 8, 6)


def plot_risk_label_heatmap(df: pd.DataFrame, plot_dir: Path) -> None:
    d = df.groupby(["split", "label_str", "risk_level"], observed=False).size().reset_index(name="n")
    d = pct_labels(d, ["split", "label_str"])
    d["text"] = d["n"].astype(str) + "\n" + d["pct"].round(1).astype(str) + "%"

    g = (
        p9.ggplot(d, p9.aes("risk_level", "label_str", fill="pct"))
        + p9.geom_tile(color="white", size=0.7)
        + p9.geom_text(p9.aes(label="text"), size=7, color=CL_CHARCOAL)
        + p9.facet_wrap("~split")
        + p9.scale_fill_gradient(low="#f5f5f5", high=CL_RED)
        + p9.labs(
            title="Risk-level concentration by label",
            subtitle="Cell color shows percentage inside each label/split group",
            x="Risk level",
            y="Label",
            fill="%",
        )
        + theme_thesis(8, 3.5)
    )
    save_plot(g, plot_dir / "risk_label_heatmap", 8, 6)


def plot_top_categories(cat_df: pd.DataFrame, plot_dir: Path) -> None:
    if cat_df.empty:
        return
    grooming = cat_df[cat_df["label_str"] == "grooming"]
    if grooming.empty:
        return
    top = (
        grooming.groupby("category", observed=False)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N_CATEGORIES)
        .index.tolist()
    )
    d = cat_df[cat_df["category"].isin(top)]
    d = d.groupby(["label_str", "category"], observed=False)["count"].sum().reset_index()
    d["category"] = pd.Categorical(d["category"], list(reversed(top)), ordered=True)

    g = (
        p9.ggplot(d, p9.aes("category", "count", fill="label_str"))
        + p9.geom_col(position=p9.position_dodge(width=0.72), width=0.65)
        + p9.coord_flip()
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.labs(
            title=f"Top {TOP_N_CATEGORIES} triggered categories",
            subtitle="Categories ranked by grooming-conversation hits",
            x="Category",
            y="Rule hits",
            fill="Label",
        )
        + theme_thesis(6.8, max(4.2, 0.22 * len(top) + 1.4))
    )
    save_plot(g, plot_dir / "top_categories", 6.8, max(4.2, 0.22 * len(top) + 1.4))


def plot_category_share(cat_df: pd.DataFrame, plot_dir: Path) -> None:
    if cat_df.empty:
        return
    d = cat_df.groupby(["label_str", "category"], observed=False)["count"].sum().reset_index()
    d["total"] = d.groupby("label_str", observed=False)["count"].transform("sum")
    d["share"] = d["count"] / d["total"].replace(0, pd.NA) * 100.0
    d = d.sort_values("share", ascending=False).groupby("label_str", observed=False).head(12)
    ordered = d.groupby("category", observed=False)["share"].max().sort_values().index.tolist()
    d["category"] = pd.Categorical(d["category"], ordered, ordered=True)

    g = (
        p9.ggplot(d, p9.aes("category", "share", fill="label_str"))
        + p9.geom_col(position=p9.position_dodge(width=0.7), width=0.62)
        + p9.coord_flip()
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.labs(
            title="Category share inside each label",
            subtitle="Top category shares; normalized separately for grooming and benign",
            x="Category",
            y="Share of rule hits (%)",
            fill="Label",
        )
        + theme_thesis(6.8, 4.8)
    )
    save_plot(g, plot_dir / "category_share_by_label", 6.8, 4.8)


def plot_hit_distribution(df: pd.DataFrame, plot_dir: Path) -> None:
    d = df.copy()
    max_hits = int(d["total_rule_hits"].max()) if not d.empty else 1
    binwidth = 1 if max_hits <= 20 else max(1, max_hits // 30)

    g = (
        p9.ggplot(d, p9.aes("total_rule_hits", fill="label_str"))
        + p9.geom_histogram(binwidth=binwidth, alpha=0.72, boundary=0, position="identity")
        + p9.facet_wrap("~split", scales="free_y")
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.labs(
            title="Total rule-hit distribution",
            subtitle="Higher mass on the right means denser heuristic evidence",
            x="Rule hits per conversation",
            y="Conversations",
            fill="Label",
        )
        + theme_thesis(8, 5)
    )
    save_plot(g, plot_dir / "total_rule_hits_distribution", 8, 6)


def plot_category_count_distribution(df: pd.DataFrame, plot_dir: Path) -> None:
    d = df.copy()
    max_cats = int(d["n_categories"].max()) if not d.empty else 1
    max_cats = max(max_cats, 1)

    g = (
        p9.ggplot(d, p9.aes("n_categories", fill="label_str"))
        + p9.geom_histogram(binwidth=1, alpha=0.72, boundary=-0.5, position="identity")
        + p9.scale_x_continuous(breaks=list(range(0, max_cats + 1)))
        + p9.facet_wrap("~split", scales="free_y")
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.labs(
            title="Unique categories per conversation",
            subtitle="Broader category coverage usually means stronger multi-signal evidence",
            x="Unique categories",
            y="Conversations",
            fill="Label",
        )
        + theme_thesis(8, 5)
    )
    save_plot(g, plot_dir / "unique_categories_distribution", 8, 6)


def plot_severity3_distribution(df: pd.DataFrame, plot_dir: Path) -> None:
    d = df.copy()
    cap = min(int(d["sev3_hits"].max()) if not d.empty else 1, 25)
    d["sev3_capped"] = d["sev3_hits"].clip(upper=cap)
    d["sev3_label"] = d["sev3_capped"].astype(str)
    if int(d["sev3_hits"].max()) > cap:
        d.loc[d["sev3_hits"] >= cap, "sev3_label"] = f"{cap}+"

    order = [str(i) for i in range(cap + 1)]
    if int(d["sev3_hits"].max()) > cap:
        order[-1] = f"{cap}+"
    d["sev3_label"] = pd.Categorical(d["sev3_label"], order, ordered=True)

    c = d.groupby(["split", "label_str", "sev3_label"], observed=False).size().reset_index(name="n")

    g = (
        p9.ggplot(c, p9.aes("sev3_label", "n", fill="label_str"))
        + p9.geom_col(position=p9.position_dodge(width=0.72), width=0.65)
        + p9.facet_wrap("~split", scales="free_y")
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.labs(
            title="Severity-3 hits per conversation",
            subtitle="Severity-3 means immediate explicit rule evidence",
            x="Severity-3 hits",
            y="Conversations",
            fill="Label",
        )
        + theme_thesis(8, 6)
    )
    save_plot(g, plot_dir / "severity3_distribution", 8, 6)


def plot_structural_signals(sig_df: pd.DataFrame, plot_dir: Path) -> None:
    if sig_df.empty:
        return
    g = (
        p9.ggplot(sig_df, p9.aes("signal", "rate", fill="label_str"))
        + p9.geom_col(position=p9.position_dodge(width=0.72), width=0.65)
        + p9.facet_wrap("~split")
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.scale_y_continuous(limits=(0, 100), expand=(0, 0, 0.04, 0))
        + p9.labs(
            title="Structural signal rates",
            subtitle="Percentage of conversations with escalation, stage arc or dominance flag",
            x="Signal",
            y="Conversations (%)",
            fill="Label",
        )
        + theme_thesis(8, 5)
    )
    save_plot(g, plot_dir / "structural_signal_rates", 8, 6)


def plot_message_volume(df: pd.DataFrame, plot_dir: Path) -> None:
    d = df.copy()

    n_messages = pd.to_numeric(d["n_messages"], errors="coerce").fillna(0.0).astype(float)
    n_pred = pd.to_numeric(d["n_pred_messages"], errors="coerce").fillna(0.0).astype(float)

    d["pred_share"] = (n_pred / n_messages.where(n_messages != 0) * 100.0).fillna(0.0)
    d["pred_share"] = pd.to_numeric(d["pred_share"], errors="coerce").fillna(0.0).astype(float)
    d = d[d["pred_share"].between(0.0, 100.0)].copy()

    g = (
        p9.ggplot(d, p9.aes(x="pred_share", fill="label_str"))
        + p9.geom_histogram(binwidth=5, alpha=0.72, boundary=0, position="identity")
        + p9.facet_wrap("~split", scales="free_y")
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.scale_x_continuous(limits=(0, 100), breaks=list(range(0, 101, 20)))
        + p9.labs(
            title="Predator-author message share",
            subtitle="Percentage of messages authored by labeled predator account",
            x="Predator-author message share (%)",
            y="Conversations",
            fill="Label",
        )
        + theme_thesis(8, 5)
    )
    save_plot(g, plot_dir / "predator_message_share", 8, 6)


def plot_fuzzy_vs_regex(method_df: pd.DataFrame, plot_dir: Path) -> None:
    if method_df.empty:
        return
    d = method_df.groupby(["label_str", "method"], observed=False)["count"].sum().reset_index()
    g = (
        p9.ggplot(d, p9.aes("method", "count", fill="label_str"))
        + p9.geom_col(position=p9.position_dodge(width=0.72), width=0.65)
        + p9.scale_fill_manual(values=LABEL_COLORS)
        + p9.labs(
            title="Regex vs fuzzy rule hits",
            subtitle="Only available when rule tags include the method field",
            x="Rule matching method",
            y="Rule hits",
            fill="Label",
        )
        + theme_thesis(5.9, 3.4)
    )
    save_plot(g, plot_dir / "regex_vs_fuzzy_hits", 8, 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_plots(records: list[dict], out: Path) -> None:
    plot_dir = out / PLOT_DIR
    df = records_frame(records)
    cat_df = category_frame(records)
    meth_df = method_frame(records)
    sig_df = signal_frame(records)

    if df.empty:
        raise ValueError("No records to plot")

    plot_risk_distribution(df, plot_dir)
    plot_risk_label_heatmap(df, plot_dir)
    plot_top_categories(cat_df, plot_dir)
    plot_category_share(cat_df, plot_dir)
    plot_hit_distribution(df, plot_dir)
    plot_category_count_distribution(df, plot_dir)
    plot_severity3_distribution(df, plot_dir)
    plot_structural_signals(sig_df, plot_dir)
    plot_message_volume(df, plot_dir)
    plot_fuzzy_vs_regex(meth_df, plot_dir)


def print_summary(records: list[dict]) -> None:
    df = records_frame(records)
    if df.empty:
        print("no records")
        return

    print("\nsummary:")
    for split in sorted(df["split"].unique()):
        s = df[df["split"] == split]
        pos = int((s["label_str"] == "grooming").sum())
        neg = int((s["label_str"] == "benign").sum())
        high_pos = int(((s["label_str"] == "grooming") & (s["risk_level"] == "HIGH")).sum())
        high_neg = int(((s["label_str"] == "benign") & (s["risk_level"] == "HIGH")).sum())
        print(
            f"  {split:<5s} total={len(s):6d} grooming={pos:5d} benign={neg:6d} "
            f"high_grooming={high_pos:5d} high_benign={high_neg:5d}"
        )


def main() -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for split in SPLITS:
        records.extend(load_split(split))

    if not records:
        raise FileNotFoundError(f"No split JSON files found in {DATASET_DIR}")

    for split in SPLITS:
        if any(r["split"] == split for r in records):
            write_outputs(records, split, out)

    generate_plots(records, out)
    print_summary(records)
    print(f"\nsaved output in: {out}")


if __name__ == "__main__":
    main()
