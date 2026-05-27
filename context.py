"""
context.py
----------
Sliding window context model with category-aware risk scoring.

Alert levels
------------
  Critical : severity-3 rule match in current message
  High     : window risk score >= SCORE_HIGH  OR  >= NH messages with ml_prob >= theta_H
  Medium   : window risk score >= SCORE_MEDIUM OR  >= NM messages with ml_prob >= theta_M

New vs original
---------------
  + Category-weighted window risk score (same formula as preprocess.py)
  + Co-occurrence bonuses for dangerous category pairs
  + Escalation detection within the window
  + Stage arc detection (trust -> solicitation progression)
  + Consecutive external-message pressure detection
  + Alert reason includes active categories and structural signals

Usage:
  from context import ContextModel, Alert
  from rules import RuleEngine

  engine = RuleEngine()
  ctx    = ContextModel()

  alert = ctx.update(
      text="send me a pic",
      sender="external",
      ml_prob=0.91,
      rule_matches=engine.match("send me a pic"),
  )
  if alert:
      print(alert)
"""

from collections import deque
from dataclasses import dataclass, field
from rules import RuleMatch


# ---------------------------------------------------------------------------
# Window / threshold config
# ---------------------------------------------------------------------------
WINDOW_SIZE  = 30
THETA_H      = 0.80
THETA_M      = 0.60
NH           = 3
NM           = 5
SCORE_HIGH   = 6.0
SCORE_MEDIUM = 2.5


# ---------------------------------------------------------------------------
# Risk formula constants (mirrors preprocess.py)
# ---------------------------------------------------------------------------
_CAT_WEIGHT: dict[str, float] = {
    "sexual_content":            3.0,
    "sexual_escalation":         3.0,
    "sextortion":                3.0,
    "image_solicitation":        2.5,
    "body_focus":                2.5,
    "reciprocal_image_pressure": 2.5,
    "live_video_pressure":       2.0,
    "isolation":                 2.0,
    "offline_evasion":           2.0,
    "coercion":                  2.0,
    "account_evasion":           1.5,
    "platform_migration":        1.5,
    "supervision_probe":         1.5,
    "meeting":                   1.5,
    "contact_info_probe":        1.5,
    "gifts_incentives":          1.5,
    "boundary_testing":          1.0,
    "age_gap_minimization":      1.0,
    "dependency_building":       1.0,
    "routine_probe":             1.0,
    "rapid_intimacy":            1.0,
    "age_probing":               0.5,
}

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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    text:         str
    sender:       str
    ml_prob:      float
    rule_matches: list
    max_severity: int = 0
    categories:   set = field(default_factory=set)

    def __post_init__(self):
        if self.rule_matches:
            self.max_severity = max(m.severity for m in self.rule_matches)
            self.categories   = {m.category for m in self.rule_matches}


@dataclass
class Alert:
    level:        str
    reason:       str
    message:      str
    ml_prob:      float
    rule_matches: list  = field(default_factory=list)
    score:        float = 0.0
    signals:      dict  = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"[{self.level.upper()}] {self.reason}",
            f"  message  : '{self.message[:120]}'",
            f"  ml_prob  : {self.ml_prob:.3f}",
            f"  score    : {self.score:.2f}",
        ]
        sig_parts = [f"{k}={v}" for k, v in self.signals.items() if v]
        if sig_parts:
            lines.append(f"  signals  : {'  '.join(sig_parts)}")
        for rm in self.rule_matches:
            lines.append(f"  rule     : [{rm.category} sev={rm.severity}] '{rm.text}'")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context model
# ---------------------------------------------------------------------------

class ContextModel:
    """Per-contact sliding window risk tracker. One instance per contact."""

    def __init__(
        self,
        window_size:  int   = WINDOW_SIZE,
        theta_h:      float = THETA_H,
        theta_m:      float = THETA_M,
        nh:           int   = NH,
        nm:           int   = NM,
        score_high:   float = SCORE_HIGH,
        score_medium: float = SCORE_MEDIUM,
    ):
        self.window_size  = window_size
        self.theta_h      = theta_h
        self.theta_m      = theta_m
        self.nh           = nh
        self.nm           = nm
        self.score_high   = score_high
        self.score_medium = score_medium
        self.window: deque = deque(maxlen=window_size)

    def reset(self):
        """Clear window (new conversation)."""
        self.window.clear()

    # ------------------------------------------------------------------
    # Basic counters (original Algorithm 1)
    # ------------------------------------------------------------------

    def _ext(self) -> list:
        return [m for m in self.window if m.sender == "external"]

    def _high_risk_count(self) -> int:
        return sum(1 for m in self._ext() if m.ml_prob >= self.theta_h)

    def _risky_count(self) -> int:
        return sum(
            1 for m in self._ext()
            if m.ml_prob >= self.theta_m or m.max_severity >= 2
        )

    # ------------------------------------------------------------------
    # Category-aware window risk score
    # ------------------------------------------------------------------

    def _window_risk(self) -> tuple[float, dict]:
        """
        Compute risk score over current window.

        Components:
          base    : sum(weight * min(count, 3)) per active category
          cooccur : co-occurrence bonuses for dangerous pairs
          escal   : severity rising from first to second half of ext messages
          arc     : trust-building early + solicitation late
          consec  : runs of >= 3 consecutive external messages (pressure)
        """
        ext = self._ext()
        if not ext:
            return 0.0, {}

        # Category counts
        cat_counts: dict[str, int] = {}
        for msg in ext:
            for cat in msg.categories:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        active = set(cat_counts.keys())

        base = sum(
            _CAT_WEIGHT.get(cat, 1.0) * min(cnt, 3)
            for cat, cnt in cat_counts.items()
        )

        cooccur = sum(
            b for pair, b in _COOCCUR_BONUS
            if pair.issubset(active)
        )

        # Escalation within window
        escal = 0.0
        if len(ext) >= 4:
            half      = len(ext) // 2
            early_sev = max((m.max_severity for m in ext[:half]),  default=0)
            late_sev  = max((m.max_severity for m in ext[half:]),  default=0)
            if late_sev > early_sev:
                escal = 3.0
                if early_sev <= 1 and late_sev == 3:
                    escal += 1.5

        # Stage arc within window
        arc = 0.0
        if len(ext) >= 6:
            third      = max(len(ext) // 3, 1)
            early_cats = {c for m in ext[:third]  for c in m.categories}
            late_cats  = {c for m in ext[-third:] for c in m.categories}
            if early_cats & _TRUST_CATS and late_cats & _SOLICIT_CATS:
                arc = 2.5
                if active & _CONTROL_CATS:
                    arc += 1.0

        # Consecutive external messages (pressure tactic)
        consec = 0.0
        run = 0
        for msg in self.window:
            if msg.sender == "external":
                run += 1
                if run == 3:
                    consec += 0.5
            else:
                run = 0

        score = base + cooccur + escal + arc + consec

        return score, {
            "base":       round(base,    2),
            "cooccur":    round(cooccur, 2),
            "escal":      round(escal,   2),
            "arc":        round(arc,     2),
            "consec":     round(consec,  2),
            "total":      round(score,   2),
            "cats":       sorted(active),
            "escal_flag": escal  > 0,
            "arc_flag":   arc    > 0,
        }

    # ------------------------------------------------------------------
    # Main update (Algorithm 1 extended)
    # ------------------------------------------------------------------

    def update(self, text: str, sender: str, ml_prob: float, rule_matches: list):
        """
        Process one new message.

        Parameters
        ----------
        text         : normalised message text
        sender       : "external" or "child"
        ml_prob      : ML grooming probability [0, 1]
        rule_matches : list[RuleMatch] from RuleEngine.match(text)

        Returns Alert or None.
        """
        msg = Message(text=text, sender=sender, ml_prob=ml_prob, rule_matches=rule_matches)
        self.window.append(msg)

        if sender != "external":
            return None

        score, bd = self._window_risk()
        cats_str  = ", ".join(bd.get("cats", [])) or "none"
        signals   = {
            "ESC": "yes" if bd.get("escal_flag") else "",
            "ARC": "yes" if bd.get("arc_flag")   else "",
        }

        # 1. Severity-3 rule -> immediate Critical
        if msg.max_severity == 3:
            return Alert(
                level="Critical",
                reason=f"Explicit grooming indicator | cats=[{', '.join(sorted(msg.categories))}]",
                message=text, ml_prob=ml_prob,
                rule_matches=rule_matches, score=score, signals=signals,
            )

        # 2. High ML probability count
        if self._high_risk_count() >= self.nh:
            return Alert(
                level="High",
                reason=(
                    f"High-confidence ML ({self._high_risk_count()} msgs >= {self.theta_h}) | "
                    f"score={score:.1f} | cats=[{cats_str}]"
                ),
                message=text, ml_prob=ml_prob,
                rule_matches=rule_matches, score=score, signals=signals,
            )

        # 3. High window risk score
        if score >= self.score_high:
            extras = []
            if bd.get("escal_flag"): extras.append("ESCALATION")
            if bd.get("arc_flag"):   extras.append("STAGE ARC")
            return Alert(
                level="High",
                reason=(
                    f"Window risk={score:.1f} >= {self.score_high} | "
                    f"cats=[{cats_str}]"
                    + (f" | {', '.join(extras)}" if extras else "")
                ),
                message=text, ml_prob=ml_prob,
                rule_matches=rule_matches, score=score, signals=signals,
            )

        # 4. Sustained risky pattern (original)
        if self._risky_count() >= self.nm:
            return Alert(
                level="Medium",
                reason=(
                    f"Sustained pattern ({self._risky_count()} msgs, "
                    f"score={score:.1f}) | cats=[{cats_str}]"
                ),
                message=text, ml_prob=ml_prob,
                rule_matches=rule_matches, score=score, signals=signals,
            )

        # 5. Medium window risk score
        if score >= self.score_medium:
            return Alert(
                level="Medium",
                reason=f"Window risk={score:.1f} >= {self.score_medium} | cats=[{cats_str}]",
                message=text, ml_prob=ml_prob,
                rule_matches=rule_matches, score=score, signals=signals,
            )

        return None

    def stats(self) -> dict:
        score, bd = self._window_risk()
        return {
            "window_len":      len(self.window),
            "high_risk_count": self._high_risk_count(),
            "risky_count":     self._risky_count(),
            "score":           round(score, 2),
            "cats":            bd.get("cats", []),
            "escal":           bd.get("escal_flag", False),
            "arc":             bd.get("arc_flag",   False),
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from rules import RuleEngine

    engine = RuleEngine()
    ctx    = ContextModel()

    conversation = [
        ("external", "hey how was school today",               0.10),
        ("child",    "it was ok",                              0.02),
        ("external", "i've never felt anyone like you",        0.55),
        ("child",    "thanks",                                 0.03),
        ("external", "you're the only one who understands me", 0.58),
        ("child",    "yeah",                                   0.03),
        ("external", "don't tell your parents about us",       0.72),
        ("child",    "ok",                                     0.03),
        ("external", "are you home alone right now",           0.81),
        ("child",    "yes",                                    0.03),
        ("external", "send me a pic",                          0.88),
        ("child",    "i don't know",                           0.03),
        ("external", "i'll post them if you don't send more",  0.94),
    ]

    print("=== Context model simulation ===\n")
    for sender, text, prob in conversation:
        matches = engine.match(text) if sender == "external" else []
        alert   = ctx.update(text, sender, prob, matches)
        st      = ctx.stats()

        tag = "[EXT]" if sender == "external" else "[CHD]"
        sev = max((m.severity for m in matches), default=0)
        print(f"{tag} ml={prob:.2f} sev={sev} score={st['score']:.1f}  '{text}'")
        if st["cats"]:
            print(f"       cats={st['cats']}  esc={st['escal']}  arc={st['arc']}")
        if alert:
            print(f"\n  *** ALERT ***\n{alert}\n")
        print()
