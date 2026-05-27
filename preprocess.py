"""
preprocess.py  --  PAN 2012 actual file structure
--------------------------------------------------
data/train/
  pan12-sexual-predator-identification-training-corpus-2012-05-01.xml
  pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt
data/test/
  pan12-sexual-predator-identification-test-corpus-2012-05-17.xml
  pan12-sexual-predator-identification-groundtruth-problem1.txt   <- predatory authors
  pan12-sexual-predator-identification-groundtruth-problem2.txt   <- predatory lines (unused here)

Predator .txt format  : one author ID per line
Problem1 .txt format  : conversationId<TAB>authorId  (one per line)

Output: HuggingFace DatasetDict (train/val/test), each record:
  { "conversation_id": str, "text": str, "label": int }

Usage:
  python preprocess.py

"""
import re
import unicodedata
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import json
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).parent))
from rules import RuleEngine as _RuleEngine
from sklearn.model_selection import train_test_split

_RULE_ENGINE = _RuleEngine()

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

SLANG_MAP = {
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\br\b": "are",
    r"\by\b": "why",

    r"\bpls\b": "please",
    r"\bplz\b": "please",
    r"\bthx\b": "thanks",
    r"\bthanx\b": "thanks",
    r"\bty\b": "thank you",
    r"\bnp\b": "no problem",
    r"\bsry\b": "sorry",
    r"\bsrry\b": "sorry",

    r"\bidk\b": "i do not know",
    r"\bikr\b": "i know right",
    r"\bimo\b": "in my opinion",
    r"\bimho\b": "in my humble opinion",
    r"\bngl\b": "not gonna lie",
    r"\btbh\b": "to be honest",
    r"\bnvm\b": "never mind",
    r"\bsmh\b": "shaking my head",
    r"\bfyi\b": "for your information",

    r"\blol\b": "laughing out loud",
    r"\blmao\b": "laughing my ass off",
    r"\brofl\b": "rolling on the floor laughing",
    r"\bomg\b": "oh my god",
    r"\bomw\b": "on my way",

    r"\bbrb\b": "be right back",
    r"\bgtg\b": "got to go",
    r"\bg2g\b": "got to go",
    r"\bttyl\b": "talk to you later",
    r"\bcya\b": "see you",
    r"\bcu\b": "see you",
    r"\bwb\b": "welcome back",
    r"\bafk\b": "away from keyboard",

    r"\bbtw\b": "by the way",
    r"\bafaik\b": "as far as i know",
    r"\basap\b": "as soon as possible",
    r"\birl\b": "in real life",
    r"\bidc\b": "i do not care",

    r"\bbf\b": "boyfriend",
    r"\bgf\b": "girlfriend",
    r"\bbff\b": "best friend forever",
    r"\bcrush\b": "romantic interest",

    r"\bima\b": "i am going to",
    r"\bimma\b": "i am going to",
    r"\bgonna\b": "going to",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "got to",
    r"\blemme\b": "let me",
    r"\bgimme\b": "give me",
    r"\bkinda\b": "kind of",
    r"\bsorta\b": "sort of",
    r"\boutta\b": "out of",
    r"\bcuz\b": "because",
    r"\bcoz\b": "because",
    r"\bcause\b": "because",

    r"\bwyd\b": "what are you doing",
    r"\bhmu\b": "hit me up",
    r"\bwbu\b": "what about you",
    r"\bhbu\b": "how about you",
    r"\bwydrn\b": "what are you doing right now",
    r"\brn\b": "right now",
    r"\btmr\b": "tomorrow",
    r"\btmrw\b": "tomorrow",
    r"\btonite\b": "tonight",
    r"\bl8r\b": "later",
    r"\bgr8\b": "great",
    r"\bm8\b": "mate",

    r"\basl\b": "age sex location",
    r"\bpic\b": "picture",
    r"\bpics\b": "pictures",
    r"\bvid\b": "video",
    r"\bvids\b": "videos",
    r"\bdm\b": "direct message",
    r"\bdms\b": "direct messages",
    r"\bpm\b": "private message",
    r"\bpms\b": "private messages",

    r"\bfr\b": "for real",
    r"\bfrfr\b": "for real for real",
    r"\bnoob\b": "newbie",
    r"\bn00b\b": "newbie",
    r"\bppl\b": "people",
    r"\bpeeps\b": "people",
    r"\bsup\b": "what is up",
    r"\bwat\b": "what",
    r"\bwut\b": "what",
    r"\btho\b": "though",
    r"\bthru\b": "through",

    r"\bkk\b": "okay",
    r"\bokay\b": "okay",
    r"\bok\b": "okay",
    r"\bk\b": "okay",
    r"\byeah\b": "yes",
    r"\bye\b": "yes",
    r"\byep\b": "yes",
    r"\byup\b": "yes",
    r"\bnah\b": "no",
    r"\bnope\b": "no",

    r"\bily\b": "i love you",
    r"\bilysm\b": "i love you so much",
    r"\bilu\b": "i love you",
    r"\bxoxo\b": "hugs and kisses",

    r"\bjk\b": "just kidding",
    r"\bjkng\b": "just kidding",
    r"\bjw\b": "just wondering",
    r"\bn1\b": "nice one",
    r"\bgg\b": "good game",
    r"\bwp\b": "well played",
}

_SLANG = [(re.compile(p, re.IGNORECASE), v) for p, v in SLANG_MAP.items()]

URL_RE    = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE  = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
PHONE_RE  = re.compile(r"\b\d[\d\s\-]{5,13}\d\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")
WS_RE     = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    for pat, rep in _SLANG:
        text = pat.sub(rep, text)
    text = WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Ground truth loaders
# ---------------------------------------------------------------------------

def load_predator_ids_txt(path: str) -> set:
    """One predator author ID per line. Skip blank lines and comments."""
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    print(f"  {len(ids)} predator IDs from {Path(path).name}")
    return ids


def load_problem1_gt(path: str) -> dict:
    """
    problem1.txt: conversationId<TAB>authorId
    Returns: { conversation_id: set_of_predatory_author_ids }
    """
    gt = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()  # fallback: space separator
            if len(parts) < 2:
                continue
            conv_id, author_id = parts[0].strip(), parts[1].strip()
            gt.setdefault(conv_id, set()).add(author_id)
    return gt


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------

def parse_xml(
    xml_path: str,
    predator_ids: Optional[set] = None,
    test_gt1: Optional[dict] = None,
) -> list:
    """
    Parse conversations XML.

    Labelling:
      train: label=1 if any message author in predator_ids
      test:  label=1 if conversation_id in test_gt1

    Provide exactly one of predator_ids or test_gt1.
    """
    assert (predator_ids is None) != (test_gt1 is None)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []

    for conv in root.iter("conversation"):
        conv_id = conv.get("id", "").strip()
        if predator_ids is not None:
            local_pred = predator_ids
        else:
            assert test_gt1 is not None
            local_pred = test_gt1.get(conv_id, set())

        messages = []
        is_grooming = False

        for msg in conv.iter("message"):
            author   = (msg.findtext("author") or "").strip()
            text_raw = (msg.findtext("text")   or "").strip()
            is_pred  = author in local_pred
            if is_pred:
                is_grooming = True
            messages.append({"author": author, "text_raw": text_raw, "is_pred": is_pred})

        records.append({
            "conversation_id": conv_id,
            "text":  flatten(messages),
            "label": int(is_grooming),
        })

    return records


# Risk category weights for the conversation-level risk formula.
# Higher weight = more grooming-specific signal when co-occurring with others.
_CAT_WEIGHT: dict[str, float] = {
    "sexual_content":           3.0,
    "sexual_escalation":        3.0,
    "sextortion":               3.0,
    "image_solicitation":       2.5,
    "body_focus":               2.5,
    "reciprocal_image_pressure": 2.5,
    "live_video_pressure":      2.0,
    "isolation":                2.0,
    "offline_evasion":          2.0,
    "coercion":                 2.0,
    "account_evasion":          1.5,
    "platform_migration":       1.5,
    "supervision_probe":        1.5,
    "meeting":                  1.5,
    "contact_info_probe":       1.5,
    "gifts_incentives":         1.5,
    "boundary_testing":         1.0,
    "age_gap_minimization":     1.0,
    "dependency_building":      1.0,
    "routine_probe":            1.0,
    "rapid_intimacy":           1.0,
    "age_probing":              0.5,
}

# Co-occurrence bonus: pairs that together raise risk substantially
_COOCCUR_BONUS: list[tuple[set, float]] = [
    ({"coercion",    "sexual_content"},           2.0),
    ({"coercion",    "sexual_escalation"},        2.0),
    ({"coercion",    "image_solicitation"},       1.5),
    ({"sextortion",  "image_solicitation"},       2.0),
    ({"sextortion",  "coercion"},                 2.0),
    ({"isolation",   "sexual_content"},           1.5),
    ({"isolation",   "image_solicitation"},       1.5),
    ({"meeting",     "supervision_probe"},        1.5),
    ({"meeting",     "offline_evasion"},          2.0),
    ({"contact_info_probe",  "meeting"},          1.5),
    ({"gifts_incentives",    "image_solicitation"}, 1.5),
    ({"platform_migration",  "sexual_content"},   1.5),
    ({"dependency_building", "image_solicitation"}, 1.5),
    ({"dependency_building", "isolation"},        1.5),
    ({"boundary_testing",    "sexual_escalation"}, 2.0),
    ({"boundary_testing",    "image_solicitation"}, 1.5),
    ({"rapid_intimacy",      "image_solicitation"}, 1.5),
    ({"rapid_intimacy",      "sexual_content"},   2.0),
    ({"gifts_incentives",    "supervision_probe"}, 1.5),
    ({"account_evasion",     "image_solicitation"}, 1.5),
    ({"account_evasion",     "sexual_content"},   1.5),
]

# Grooming stage categories (from Craven et al. framework)
# Trust-building comes before solicitation in classic grooming arc
_TRUST_CATS = {
    "rapid_intimacy", "dependency_building", "age_gap_minimization",
    "boundary_testing", "gifts_incentives",
}
_SOLICIT_CATS = {
    "image_solicitation", "sexual_content", "sexual_escalation",
    "body_focus", "sextortion", "reciprocal_image_pressure",
}
_CONTROL_CATS = {
    "coercion", "isolation", "account_evasion", "offline_evasion",
    "supervision_probe", "platform_migration",
}

# Thresholds
_RISK_HIGH      = 6.0
_RISK_MEDIUM    = 2.5
_ESCAL_BONUS    = 3.0   # severity increases in second half of conversation
_STAGE_ARC_BONUS = 2.5  # trust-building -> solicitation arc detected
_CONSEC_BONUS   = 0.5   # per run of 3+ consecutive PRED messages (pressure)
_DOMINANCE_BONUS = 1.5  # PRED sends >60% of messages
_DENSITY_BONUS  = 2.0   # >1.5 rule hits per PRED message on average


def _conv_risk_score(
    cat_counts:    dict[str, int],
    pred_msgs:     list[dict],   # list of {norm, matches, idx_in_conv}
    total_msgs:    int,
) -> tuple[float, str, dict]:
    """
    Compute conversation-level risk score.

    Components
    ----------
    1. Base score     : sum(weight * min(count, 3)) per category
    2. Co-occurrence  : bonuses for dangerous category pairs
    3. Escalation     : severity rises from first half to second half
    4. Stage arc      : trust-building categories present before solicitation
    5. Dominance      : predator sends >60% of messages
    6. Density        : high rule-hit rate per predator message
    7. Consecutive    : runs of 3+ predator messages without child response

    Returns (score, level, breakdown_dict).
    """
    active = {cat for cat, cnt in cat_counts.items() if cnt > 0}
    n_pred = len(pred_msgs)

    # 1. Base
    base = sum(
        _CAT_WEIGHT.get(cat, 1.0) * min(cnt, 3)
        for cat, cnt in cat_counts.items()
    )

    # 2. Co-occurrence
    cooccur = sum(b for pair, b in _COOCCUR_BONUS if pair.issubset(active))

    # 3. Escalation: compare max severity in first vs second half of pred messages
    escal = 0.0
    if n_pred >= 4:
        half      = n_pred // 2
        early_sev = max((m["max_sev"] for m in pred_msgs[:half]),  default=0)
        late_sev  = max((m["max_sev"] for m in pred_msgs[half:]),  default=0)
        if late_sev > early_sev:
            escal = _ESCAL_BONUS
            # Extra if escalation crosses severity boundary (e.g. 1->3)
            if early_sev <= 1 and late_sev == 3:
                escal += 1.5

    # 4. Stage arc: trust cats in first third, solicit cats in last third
    arc = 0.0
    if n_pred >= 6:
        third     = max(n_pred // 3, 1)
        early_cats = {cat for m in pred_msgs[:third]  for cat in m["cats"]}
        late_cats  = {cat for m in pred_msgs[-third:] for cat in m["cats"]}
        has_trust_early   = bool(early_cats & _TRUST_CATS)
        has_solicit_late  = bool(late_cats  & _SOLICIT_CATS)
        has_control_any   = bool(active     & _CONTROL_CATS)
        if has_trust_early and has_solicit_late:
            arc = _STAGE_ARC_BONUS
            if has_control_any:
                arc += 1.0  # full grooming arc: trust -> control -> solicitation

    # 5. Message dominance
    dominance = 0.0
    if total_msgs > 0 and n_pred / total_msgs > 0.60:
        dominance = _DOMINANCE_BONUS

    # 6. Rule density: total rule hits across all pred messages
    total_hits = sum(len(m["matches"]) for m in pred_msgs)
    density = 0.0
    if n_pred > 0 and total_hits / n_pred > 1.5:
        density = _DENSITY_BONUS

    # 7. Consecutive pred messages (runs of >= 3 without child response)
    consec = 0.0
    run = 0
    # pred_msgs carry their original conversation index; gaps = child messages
    for i, m in enumerate(pred_msgs):
        if i == 0:
            run = 1
            continue
        if m["conv_idx"] == pred_msgs[i-1]["conv_idx"] + 1:
            run += 1
            if run == 3:
                consec += _CONSEC_BONUS
        else:
            run = 1

    score = base + cooccur + escal + arc + dominance + density + consec

    if score >= _RISK_HIGH:
        level = "HIGH"
    elif score >= _RISK_MEDIUM:
        level = "MEDIUM"
    else:
        level = "LOW"

    breakdown = {
        "base": round(base, 2),
        "cooccur": round(cooccur, 2),
        "escal": round(escal, 2),
        "arc": round(arc, 2),
        "dominance": round(dominance, 2),
        "density": round(density, 2),
        "consec": round(consec, 2),
        "total": round(score, 2),
    }
    return score, level, breakdown


def flatten(messages: list) -> str:
    """
    Build role-tagged, rule-annotated, normalised conversation string.

    Prefix tokens (at [CLS] position):
      [RISK:HIGH] [CATS:isolation,sexual_content] [ESC:1] [ARC:1] [DOM:1]

    ESC=1  -> severity escalated in second half
    ARC=1  -> trust-building -> solicitation arc detected
    DOM=1  -> predator dominates message count (>60%)

    Per-message rule tags (PRED messages only):
      [PRED] [RULE:image_solicitation:3] [RULE:isolation:2] send me a pic
    """
    cat_counts: dict[str, int] = {}
    pred_msgs:  list[dict]     = []
    parts:      list[str]      = []
    total_msgs  = 0

    for conv_idx, msg in enumerate(messages):
        norm = normalise(msg["text_raw"])
        if not norm:
            continue

        total_msgs += 1
        role = "[PRED]" if msg["is_pred"] else "[USER]"

        if msg["is_pred"]:
            matches  = _RULE_ENGINE.match(norm)
            max_sev  = max((m.severity for m in matches), default=0)
            cats_hit = {m.category for m in matches}

            # Update category counts
            for cat in cats_hit:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

            # Store for structural analysis
            pred_msgs.append({
                "conv_idx": conv_idx,
                "max_sev":  max_sev,
                "cats":     cats_hit,
                "matches":  matches,
            })

            # Build rule tags (deduplicated per category+severity)
            seen: set = set()
            tags: list[str] = []
            for m in matches:
                key = (m.category, m.severity)
                if key not in seen:
                    seen.add(key)
                    tags.append(f"[RULE:{m.category}:{m.severity}]")
            rule_prefix = (" ".join(tags) + " ") if tags else ""
            parts.append(f"{role} {rule_prefix}{norm}")
        else:
            parts.append(f"{role} {norm}")

    # Compute full risk score
    score, level, bd = _conv_risk_score(cat_counts, pred_msgs, total_msgs)

    # Structural signal tokens
    esc_tok = "[ESC:1]" if bd["escal"] > 0 else "[ESC:0]"
    arc_tok = "[ARC:1]" if bd["arc"]   > 0 else "[ARC:0]"
    dom_tok = "[DOM:1]" if bd["dominance"] > 0 else "[DOM:0]"

    active_cats = sorted(cat_counts.keys())
    cats_str    = ",".join(active_cats) if active_cats else "none"
    prefix      = f"[RISK:{level}] [CATS:{cats_str}] {esc_tok} {arc_tok} {dom_tok}"

    body = " [SEP] ".join(parts)
    return f"{prefix} {body}" if body else prefix


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def to_records(records: list) -> list:
    return [
        {"conversation_id": r["conversation_id"], "text": r["text"], "label": r["label"]}
        for r in records
    ]


def print_stats(ds: dict) -> None:
    for split, data in ds.items():
        n   = len(data)
        pos = sum(r["label"] for r in data)
        print(f"  {split:6s}: {n:6d} total | {pos:5d} pos ({100*pos/n:.1f}%) | {n-pos:5d} neg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAIN_XML  = "data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
TRAIN_PRED = "data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
TEST_XML   = "data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"
TEST_GT1   = "data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt"
OUTPUT     = "pan12_dataset"
VAL_RATIO  = 0.1
SEED       = 42
# ---------------------------------------------------------------------------


def main():
    # --- Training ---
    print("Training predator IDs...")
    pred_ids = load_predator_ids_txt(TRAIN_PRED)

    print("Parsing training XML...")
    train_all = parse_xml(TRAIN_XML, predator_ids=pred_ids)
    pos = sum(r["label"] for r in train_all)
    print(f"  {len(train_all)} conversations | {pos} positive ({100*pos/len(train_all):.1f}%)")

    labels = [r["label"] for r in train_all]
    train_rec, val_rec = train_test_split(
        train_all, test_size=VAL_RATIO, stratify=labels, random_state=SEED
    )

    # --- Test ---
    print("Loading test ground truth (problem1)...")
    gt1 = load_problem1_gt(TEST_GT1)
    print(f"  {len(gt1)} predatory conversations in problem1")

    print("Parsing test XML...")
    test_rec = parse_xml(TEST_XML, test_gt1=gt1)
    pos_t = sum(r["label"] for r in test_rec)
    print(f"  {len(test_rec)} conversations | {pos_t} positive ({100*pos_t/len(test_rec):.1f}%)")

    # --- Save ---
    ds = {
        "train": to_records(train_rec),
        "val":   to_records(val_rec),
        "test":  to_records(test_rec),
    }

    print("\nDataset statistics:")
    print_stats(ds)

    out = Path(OUTPUT)
    out.mkdir(parents=True, exist_ok=True)
    for split, records in ds.items():
        path = out / f"{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  {split}.json  ({path.stat().st_size // 1024} KB)")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
