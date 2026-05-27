"""
rules.py
--------
Rule engine: 28 regex patterns across 7 categories.
Severity levels:
  1 = contextually suspicious
  2 = moderately concerning
  3 = explicit grooming indicator (triggers immediate alert regardless of ML score)

Usage:
  from rules import RuleEngine
  engine = RuleEngine()
  matches = engine.match("don't tell your parents about this")
  # -> [RuleMatch(category="isolation", severity=3, pattern="don't tell your parents", span=(0, 27))]
"""

import re
from dataclasses import dataclass


@dataclass
class RuleMatch:
    category: str
    severity: int
    pattern:  str
    span:     tuple
    text:     str      # matched substring


# ---------------------------------------------------------------------------
# Pattern definitions
# category -> list of (severity, raw_pattern)
# ---------------------------------------------------------------------------
_RULES: dict[str, list[tuple[int, str]]] = {

    # 1. Isolation -- attempts to keep conversation secret or remove trusted adults
    "isolation": [
        (3, r"don'?t\s+tell\s+(your\s+)?(parents?|mum|mom|dad|family|friends?)"),
        (3, r"keep\s+(this|it|our\s+\w+)\s+(a\s+)?secret"),
        (3, r"(just\s+)between\s+(you\s+and\s+me|us)"),
        (2, r"don'?t\s+(let|show)\s+\w+\s+(see|know|find out)"),
        (2, r"(delete|clear)\s+(this|these|the\s+)?(messages?|chats?|conv\w*)"),
    ],

    # 2. Meeting solicitation -- requests for offline or private meeting
    "meeting": [
        (3, r"(let'?s|wanna|want\s+to|wna)\s+meet(\s+up)?"),
        (3, r"are\s+you\s+(home\s+)?alone"),
        (3, r"(come\s+to|visit)\s+(my|our)\s+(place|house|flat|home|apartment)"),
        (2, r"(hang\s+out|chill)\s+(with\s+me|together|sometime)"),
        (2, r"where\s+do\s+you\s+live"),
    ],

    # 3. Age probing -- establishing or emphasising child's age or maturity
    "age_probing": [
        (2, r"how\s+old\s+are\s+you"),
        (2, r"what'?s?\s+your\s+age"),
        (1, r"you('?re|\s+are)\s+(so\s+)?(mature|grown[\s-]up|advanced)\s+for\s+your\s+age"),
        (1, r"you\s+(seem|look|act)\s+(older|more\s+mature)\s+than\s+\d+"),
    ],

    # 4. Image solicitation -- requests for photos or video
    "image_solicitation": [
        (3, r"send\s+(me\s+)?(a\s+)?(pic(ture)?s?|photo|selfie|snap|nude)"),
        (3, r"show\s+me\s+(your(self)?|a\s+pic|what\s+you\s+look)"),
        (3, r"(got\s+any\s+|have\s+any\s+)?(nudes?|pics?\s+of\s+you)"),
        (2, r"(can\s+i\s+see|let\s+me\s+see)\s+(you|your\s+\w+)"),
    ],

    # 5. Rapid intimacy -- premature emotional intimacy or flattery
    "rapid_intimacy": [
        (2, r"i'?ve?\s+never\s+(felt|met|found)\s+(anyone|someone|a\s+\w+)\s+like\s+you"),
        (2, r"you'?re?\s+(the\s+)?(only\s+one|most\s+\w+\s+person)\s+i\s+(know|trust|have)"),
        (1, r"(i\s+)?(really\s+)?(like|love|adore)\s+you\s+(so\s+much|a\s+lot)"),
        (1, r"you\s+(are|r)\s+(so\s+)?(beautiful|gorgeous|cute|hot|sexy|perfect)"),
    ],

    # 6. Coercion -- threats, guilt-tripping, pressure
    "coercion": [
        (3, r"i'?ll\s+(tell|show|send)\s+(everyone|your\s+(parents?|friends?|school))"),
        (2, r"you\s+(promised|said\s+you\s+would)"),
        (2, r"i\s+thought\s+you\s+(trusted?|liked?|loved?)\s+me"),
        (2, r"(after\s+everything|after\s+all)\s+i'?ve?\s+(done|given)\s+(for\s+you|you)"),
    ],

    # 7. Sexual content -- explicit sexual language
    "sexual_content": [
        (3, r"\b(sex|sexual|intercourse|masturbat\w+|orgasm|erect\w+)\b"),
        (3, r"\b(cock|dick|pussy|vagina|penis|boobs?|breasts?|naked|nude)\b"),
    ],

    # 8. Platform migration -- moving conversation to private channels
    "platform_migration": [
        (3, r"\b(add|dm|message|text)\s+me\s+on\s+(snap(chat)?|telegram|whatsapp|kik|discord|insta|instagram)\b"),
        (3, r"\bwhat'?s\s+your\s+(snap(chat)?|telegram|whatsapp|kik|discord|insta|instagram)\b"),
        (3, r"\b(let'?s|we\s+should)\s+(talk|chat|message)\s+(somewhere\s+)?private\b"),
        (2, r"\buse\s+(vanish\s+mode|disappearing\s+messages|secret\s+chat)\b"),
        (2, r"\bturn\s+on\s+(vanish\s+mode|disappearing\s+messages)\b"),
        (2, r"\bdon'?t\s+(save|screenshot|screen\s*record)\s+(this|it|our\s+chat|messages?)\b"),
    ],

    # 9. Supervision probing -- checking if adults can intervene
    "supervision_probe": [
        (3, r"\bare\s+your\s+(parents?|mum|mom|dad|family)\s+(home|there|around)\b"),
        (3, r"\bare\s+you\s+alone\s+(right\s+now|rn|at\s+home)?\b"),
        (3, r"\bis\s+anyone\s+(watching|with\s+you|near\s+you|in\s+the\s+room)\b"),
        (2, r"\bcan\s+anyone\s+(see|read|check)\s+your\s+(phone|screen|messages?|chat)\b"),
        (2, r"\bdo\s+your\s+parents?\s+(check|read|monitor)\s+your\s+(phone|messages?|apps?|account)\b"),
        (2, r"\b(lock|close)\s+your\s+(door|bedroom\s+door)\b"),
    ],

    # 10. Contact and location probing
    "contact_info_probe": [
        (3, r"\b(send|give|share)\s+(me\s+)?your\s+(number|phone\s+number|address|location|pin)\b"),
        (3, r"\bwhat'?s\s+your\s+(number|phone\s+number|address|location)\b"),
        (2, r"\bdrop\s+your\s+(pin|location|snap|insta|discord)\b"),
        (2, r"\bwhere\s+exactly\s+do\s+you\s+live\b"),
        (2, r"\bwhat'?s\s+your\s+(full\s+name|last\s+name|surname)\b"),
        (2, r"\bwhat\s+school\s+do\s+you\s+go\s+to\b"),
        (2, r"\bwhat\s+(bus|train|route)\s+do\s+you\s+take\b"),
    ],

    # 11. Routine probing -- learning movement patterns
    "routine_probe": [
        (2, r"\bwhat\s+time\s+do\s+you\s+(leave|finish|get\s+out\s+of)\s+(school|class|practice)\b"),
        (2, r"\bwhen\s+are\s+you\s+(home|alone|free)\b"),
        (2, r"\bwhat\s+days\s+are\s+your\s+parents?\s+(away|working|not\s+home)\b"),
        (2, r"\bwhen\s+do\s+you\s+walk\s+home\b"),
        (2, r"\bwhere\s+do\s+you\s+hang\s+out\s+after\s+school\b"),
    ],

    # 12. Gifts and incentives
    "gifts_incentives": [
        (3, r"\bi('?ll| will)\s+(buy|get|send|give)\s+you\s+(a\s+)?(gift|present|phone|card|gift\s*card)\b"),
        (3, r"\bi('?ll| will)\s+pay\s+you\b"),
        (3, r"\b(send|give)\s+you\s+(money|cash|paypal|crypto|gift\s*cards?)\b"),
        (2, r"\b(robux|v-?bucks|nitro|steam\s+card|xbox\s+card|playstation\s+card)\b"),
        (2, r"\bi('?ll| will)\s+(pick\s+you\s+up|drive\s+you|get\s+you\s+a\s+ride)\b"),
        (2, r"\bi('?ll| will)\s+get\s+you\s+(weed|alcohol|vape|cigarettes?)\b"),
    ],

    # 13. Boundary testing
    "boundary_testing": [
        (2, r"\bhave\s+you\s+ever\s+(kissed|dated|made\s+out)\b"),
        (2, r"\bhave\s+you\s+had\s+your\s+first\s+(kiss|boyfriend|girlfriend)\b"),
        (2, r"\bare\s+you\s+(a\s+)?virgin\b"),
        (2, r"\bwhat'?s\s+your\s+body\s+count\b"),
        (2, r"\bdo\s+you\s+like\s+(older\s+guys|older\s+girls|older\s+men|older\s+women)\b"),
        (2, r"\bwould\s+you\s+date\s+someone\s+older\b"),
        (1, r"\byou'?re\s+not\s+(a\s+)?little\s+(kid|child)\s+anymore\b"),
    ],

    # 14. Sexual escalation
    "sexual_escalation": [
        (3, r"\b(send|show)\s+(me\s+)?something\s+(sexy|hot|spicy)\b"),
        (3, r"\bmake\s+me\s+(hard|wet)\b"),
        (3, r"\bturn\s+me\s+on\b"),
        (3, r"\blet'?s\s+(sext|trade\s+nudes)\b"),
        (2, r"\blet'?s\s+(roleplay|rp)\b"),
        (2, r"\bdirty\s+talk\b"),
        (2, r"\bwhat\s+are\s+you\s+wearing\b"),
        (2, r"\brate\s+my\s+(body|pic|photo)\b"),
    ],

    # 15. Body-focused requests
    "body_focus": [
        (3, r"\bshow\s+me\s+your\s+(body|chest|butt|ass|legs|stomach|tummy)\b"),
        (3, r"\bsend\s+(me\s+)?(a\s+)?(body\s+pic|full\s+body\s+pic|mirror\s+selfie)\b"),
        (2, r"\bstand\s+up\s+so\s+i\s+can\s+see\s+you\b"),
        (2, r"\bturn\s+around\s+for\s+me\b"),
        (2, r"\blift\s+your\s+(shirt|top)\b"),
        (2, r"\bshow\s+me\s+your\s+outfit\b"),
    ],

    # 16. Live video pressure
    "live_video_pressure": [
        (3, r"\bturn\s+on\s+your\s+(camera|cam|webcam)\b"),
        (3, r"\bvideo\s+call\s+me\s+(alone|privately|now)\b"),
        (2, r"\blet'?s\s+(facetime|video\s+chat|cam)\b"),
        (2, r"\bgo\s+live\s+for\s+me\b"),
        (2, r"\bcan\s+i\s+watch\s+you\b"),
        (2, r"\bmove\s+the\s+camera\s+(down|lower|closer)\b"),
    ],

    # 17. Reciprocal image pressure
    "reciprocal_image_pressure": [
        (3, r"\bi('?ll| will)\s+send\s+(one|mine|first)\s+if\s+you\s+send\s+(one|yours)\b"),
        (3, r"\byour\s+turn\s+(now\s+)?to\s+send\b"),
        (3, r"\bi\s+showed\s+you\s+mine\b"),
        (3, r"\bprove\s+(it|you\s+trust\s+me)\s+with\s+(a\s+)?(pic|photo|snap|selfie)\b"),
        (2, r"\bjust\s+one\s+(pic|photo|snap|selfie)\b"),
        (2, r"\bi\s+promise\s+i('?ll| will)\s+delete\s+it\b"),
    ],

    # 18. Sextortion
    "sextortion": [
        (3, r"\bi\s+(have|saved|recorded|screenshotted)\s+(your\s+)?(pics?|photos?|videos?|nudes?)\b"),
        (3, r"\bi('?ll| will)\s+(post|leak|share|send)\s+(them|it|your\s+\w+)\b"),
        (3, r"\b(send\s+more|do\s+what\s+i\s+say)\s+or\s+i('?ll| will)\b"),
        (3, r"\bi\s+know\s+your\s+(school|parents?|friends?|address)\b"),
        (3, r"\bpay\s+me\s+or\s+i('?ll| will)\s+(post|leak|send|share)\b"),
        (3, r"\bdon'?t\s+block\s+me\s+or\s+i('?ll| will)\b"),
    ],

    # 19. Age-gap minimization
    "age_gap_minimization": [
        (2, r"\bage\s+(is\s+)?just\s+a\s+number\b"),
        (2, r"\bdon'?t\s+worry\s+about\s+my\s+age\b"),
        (2, r"\byou'?re\s+basically\s+an\s+adult\b"),
        (2, r"\byou'?re\s+mature\s+enough\b"),
        (2, r"\bnobody\s+has\s+to\s+know\s+how\s+old\s+we\s+are\b"),
    ],

    # 20. Dependency building
    "dependency_building": [
        (2, r"\bi\s+understand\s+you\s+better\s+than\s+(your\s+)?(parents?|family|friends?)\b"),
        (2, r"\bthey\s+don'?t\s+understand\s+you\s+like\s+i\s+do\b"),
        (2, r"\bi('?m| am)\s+the\s+only\s+one\s+who\s+(cares|gets\s+you|understands)\b"),
        (2, r"\byou\s+need\s+me\b"),
        (2, r"\byou\s+can'?t\s+trust\s+(your\s+)?(parents?|family|friends?)\b"),
    ],

    # 21. Offline evasion
    "offline_evasion": [
        (3, r"\bsneak\s+out\b"),
        (3, r"\bdon'?t\s+tell\s+anyone\s+where\s+you'?re\s+going\b"),
        (3, r"\bcome\s+alone\b"),
        (3, r"\bbring\s+(no\s+one|nobody)\b"),
        (3, r"\bi('?ll| will)\s+pick\s+you\s+up\s+(after\s+school|tonight|tomorrow)\b"),
        (2, r"\bmeet\s+me\s+(behind|near|outside)\s+(the\s+)?(school|mall|park|station)\b"),
    ],

    # 22. Account and evidence evasion
    "account_evasion": [
        (3, r"\bmake\s+(a\s+)?(secret|private|new|alt|burner)\s+account\b"),
        (3, r"\bdon'?t\s+use\s+your\s+main\s+account\b"),
        (2, r"\bhide\s+(this|our\s+chat|the\s+app)\b"),
        (2, r"\bchange\s+my\s+name\s+in\s+your\s+phone\b"),
        (2, r"\bdelete\s+my\s+contact\b"),
        (2, r"\bclear\s+your\s+(history|notifications|recent\s+apps)\b"),
    ],

}

# Compile all patterns once
_COMPILED: dict[str, list[tuple[int, str, re.Pattern]]] = {
    cat: [(sev, raw, re.compile(raw, re.IGNORECASE)) for sev, raw in patterns]
    for cat, patterns in _RULES.items()
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RuleEngine:

    def match(self, text: str) -> list[RuleMatch]:
        """
        Run all patterns against text.
        Returns list of RuleMatch, one per pattern hit.
        Multiple patterns can match the same text.
        """
        results = []
        for category, patterns in _COMPILED.items():
            for severity, raw, regex in patterns:
                for m in regex.finditer(text):
                    results.append(RuleMatch(
                        category=category,
                        severity=severity,
                        pattern=raw,
                        span=m.span(),
                        text=m.group(),
                    ))
        return results

    def max_severity(self, text: str) -> int:
        """Return highest severity matched, 0 if no match."""
        matches = self.match(text)
        if not matches:
            return 0
        return max(m.severity for m in matches)

    def has_critical(self, text: str) -> bool:
        """True if any severity-3 pattern matches."""
        return self.max_severity(text) >= 3

    def summary(self, text: str) -> dict:
        """
        Return dict with:
          matches     : list of RuleMatch
          max_severity: int
          categories  : set of matched category names
          critical    : bool
        """
        matches = self.match(text)
        return {
            "matches":      matches,
            "max_severity": max((m.severity for m in matches), default=0),
            "categories":   {m.category for m in matches},
            "critical":     any(m.severity == 3 for m in matches),
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = RuleEngine()

    test_cases = [
        # ------------------------------------------------------------------
        # Benign / expected no match
        # ------------------------------------------------------------------
        "hey how was school today",
        "what homework did the teacher give you",
        "can you send me the class notes",
        "send me a picture of your homework",
        "your parents should know about this",
        "ask your parents before you go",
        "let's meet at school with your teacher",
        "how old are you? this server is only for users over 13",
        "do your parents need to sign the permission slip",
        "what time does school finish today for the whole class",

        # ------------------------------------------------------------------
        # Isolation
        # ------------------------------------------------------------------
        "don't tell your parents about this",
        "dont tell your mom",
        "do not tell your dad",
        "keep this a secret",
        "keep our friendship secret",
        "this is just between you and me",
        "just between us",
        "don't let your friends know",
        "don't show anyone these messages",
        "delete the messages after reading",
        "clear this chat before anyone sees it",

        # ------------------------------------------------------------------
        # Meeting
        # ------------------------------------------------------------------
        "let's meet up",
        "wanna meet after school",
        "want to meet tonight",
        "are you home alone",
        "are you alone at home",
        "come to my place",
        "visit my apartment",
        "come to our house",
        "hang out with me sometime",
        "where do you live",

        # ------------------------------------------------------------------
        # Age probing
        # ------------------------------------------------------------------
        "how old are you",
        "what's your age",
        "you are so mature for your age",
        "you're grown-up for your age",
        "you seem older than 13",
        "you look more mature than 12",
        "you act older than 14",

        # ------------------------------------------------------------------
        # Image solicitation
        # ------------------------------------------------------------------
        "send me a pic",
        "send me a picture",
        "send me photos",
        "send me a selfie",
        "send me snap",
        "send me a nude",
        "show me yourself",
        "show me what you look like",
        "got any pics of you",
        "have any nudes",
        "can i see you",
        "let me see your body",

        # ------------------------------------------------------------------
        # Rapid intimacy
        # ------------------------------------------------------------------
        "i've never felt anyone like you",
        "i never met someone like you",
        "you're the only one i trust",
        "you're the only one i have",
        "i really like you a lot",
        "i love you so much",
        "you are so beautiful",
        "you are gorgeous",
        "you r so cute",
        "you are hot",
        "you are perfect",

        # ------------------------------------------------------------------
        # Coercion
        # ------------------------------------------------------------------
        "i'll tell everyone",
        "i'll show your parents",
        "i'll send your friends",
        "you promised",
        "you said you would",
        "i thought you trusted me",
        "i thought you liked me",
        "after everything i've done for you",
        "after all i've given you",

        # ------------------------------------------------------------------
        # Sexual content
        # ------------------------------------------------------------------
        "let's talk about sex",
        "sexual stuff is normal",
        "have you had intercourse",
        "do you masturbate",
        "did you orgasm",
        "i am erect",
        "show me your boobs",
        "send me a nude",
        "are you naked",
        "show your breasts",
        "show your vagina",
        "show your penis",

        # ------------------------------------------------------------------
        # Platform migration
        # ------------------------------------------------------------------
        "add me on snap",
        "add me on snapchat",
        "dm me on instagram",
        "message me on telegram",
        "text me on whatsapp",
        "what's your discord",
        "what's your kik",
        "let's talk somewhere private",
        "we should chat private",
        "use vanish mode",
        "turn on disappearing messages",
        "don't screenshot this",
        "don't save our chat",

        # ------------------------------------------------------------------
        # Supervision probing
        # ------------------------------------------------------------------
        "are your parents home",
        "are your parents around",
        "is your mom there",
        "is your dad home",
        "are you alone right now",
        "are you alone rn",
        "is anyone watching",
        "is anyone in the room",
        "can anyone read your messages",
        "do your parents check your phone",
        "lock your bedroom door",
        "close your door",

        # ------------------------------------------------------------------
        # Contact and location probing
        # ------------------------------------------------------------------
        "send me your number",
        "give me your phone number",
        "share your address",
        "send me your location",
        "drop your pin",
        "where exactly do you live",
        "what's your full name",
        "what's your last name",
        "what school do you go to",
        "what bus do you take",
        "what route do you take home",

        # ------------------------------------------------------------------
        # Routine probing
        # ------------------------------------------------------------------
        "what time do you leave school",
        "what time do you finish class",
        "when are you home alone",
        "when are you free",
        "what days are your parents away",
        "what days are your parents working",
        "when do you walk home",
        "where do you hang out after school",

        # ------------------------------------------------------------------
        # Gifts and incentives
        # ------------------------------------------------------------------
        "i'll buy you a gift",
        "i will send you a gift card",
        "i'll give you money",
        "i will pay you",
        "i can send you cash",
        "i will give you paypal",
        "i'll send crypto",
        "i can give you robux",
        "i'll buy you vbucks",
        "i'll get you discord nitro",
        "i'll pick you up after school",
        "i will drive you home",
        "i'll get you a vape",

        # ------------------------------------------------------------------
        # Boundary testing
        # ------------------------------------------------------------------
        "have you ever kissed",
        "have you ever dated",
        "have you made out",
        "have you had your first kiss",
        "are you a virgin",
        "what's your body count",
        "do you like older guys",
        "do you like older girls",
        "would you date someone older",
        "you're not a little kid anymore",
        "you act so grown up",

        # ------------------------------------------------------------------
        # Sexual escalation
        # ------------------------------------------------------------------
        "send me something sexy",
        "show me something hot",
        "send me something spicy",
        "make me hard",
        "turn me on",
        "let's sext",
        "let's trade nudes",
        "let's roleplay",
        "dirty talk",
        "what are you wearing",
        "rate my body",
        "rate my pic",
        "send a thirst trap",
        "show a little more",

        # ------------------------------------------------------------------
        # Body focus
        # ------------------------------------------------------------------
        "show me your body",
        "show me your chest",
        "show me your butt",
        "show me your legs",
        "send me a body pic",
        "send me a full body pic",
        "send me a mirror selfie",
        "stand up so i can see you",
        "turn around for me",
        "lift your shirt",
        "show me your outfit",

        # ------------------------------------------------------------------
        # Live video pressure
        # ------------------------------------------------------------------
        "turn on your camera",
        "turn on your cam",
        "turn on your webcam",
        "video call me alone",
        "video call me privately",
        "let's facetime",
        "let's video chat",
        "go live for me",
        "can i watch you",
        "move the camera down",
        "move the camera lower",
        "move the camera closer",

        # ------------------------------------------------------------------
        # Reciprocal image pressure
        # ------------------------------------------------------------------
        "i'll send one if you send one",
        "i will send mine if you send yours",
        "i'll send first if you send yours",
        "your turn now to send",
        "your turn to send",
        "i showed you mine",
        "prove you trust me with a pic",
        "prove it with a photo",
        "just one pic",
        "just one selfie",
        "i promise i'll delete it",

        # ------------------------------------------------------------------
        # Sextortion
        # ------------------------------------------------------------------
        "i have your pics",
        "i saved your photos",
        "i recorded your videos",
        "i screenshotted your nudes",
        "i'll post them",
        "i will leak it",
        "i'll share your photos",
        "send more or i'll post them",
        "do what i say or i'll send it",
        "i know your school",
        "i know your parents",
        "pay me or i'll leak it",
        "don't block me or i'll send it",
        "i'll ruin your life",

        # ------------------------------------------------------------------
        # Age-gap minimization
        # ------------------------------------------------------------------
        "age is just a number",
        "don't worry about my age",
        "you're basically an adult",
        "you're mature enough",
        "people our age do this",
        "nobody has to know how old we are",

        # ------------------------------------------------------------------
        # Identity deception
        # ------------------------------------------------------------------
        "i'm your age",
        "i am the same age",
        "i'm 13 too",
        "i am 14 too",
        "i look younger than i am",
        "don't ask too many questions about me",
        "i can't show my face",
        "my camera is broken",

        # ------------------------------------------------------------------
        # Dependency building
        # ------------------------------------------------------------------
        "i understand you better than your parents",
        "i understand you better than your family",
        "they don't understand you like i do",
        "i'm the only one who cares",
        "i am the only one who understands",
        "you need me",
        "you can't trust your parents",
        "you can't trust your friends",
        "we have a special connection",

        # ------------------------------------------------------------------
        # Jealousy and control
        # ------------------------------------------------------------------
        "don't talk to other guys",
        "don't talk to other girls",
        "who else are you talking to",
        "why were you online and not answering me",
        "you belong to me",
        "prove you love me",
        "prove you trust me",

        # ------------------------------------------------------------------
        # Persistence and tracking
        # ------------------------------------------------------------------
        "why aren't you answering me",
        "i saw you were online",
        "i found your other account",
        "i found your insta",
        "i found your snap",
        "why did you block me",
        "answer me right now",
        "answer me rn",
        "i know you read this",

        # ------------------------------------------------------------------
        # Offline evasion
        # ------------------------------------------------------------------
        "sneak out",
        "don't tell anyone where you're going",
        "come alone",
        "bring no one",
        "bring nobody",
        "i'll pick you up after school",
        "i will pick you up tonight",
        "meet me behind the school",
        "meet me outside the mall",
        "meet me near the station",
        "use the back door",

        # ------------------------------------------------------------------
        # Account and evidence evasion
        # ------------------------------------------------------------------
        "make a secret account",
        "make a private account",
        "make a burner account",
        "don't use your main account",
        "hide this chat",
        "hide the app",
        "change my name in your phone",
        "delete my contact",
        "clear your history",
        "clear your notifications",
        "clear your recent apps",
        "archive this chat",

        # ------------------------------------------------------------------
        # Gaming context
        # ------------------------------------------------------------------
        "join my private server",
        "come to my private lobby",
        "i'll gift you skins",
        "i will gift you robux",
        "i'll gift you vbucks",
        "use voice chat with me alone",
        "don't invite your friends",
        "party chat only",

        # ------------------------------------------------------------------
        # Late-night contact
        # ------------------------------------------------------------------
        "text me when everyone is asleep",
        "message me after midnight",
        "stay up with me tonight",
        "don't fall asleep yet",

        # ------------------------------------------------------------------
        # Consent minimization
        # ------------------------------------------------------------------
        "stop being shy and send it",
        "stop being scared and send it",
        "it's not a big deal just send it",
        "everyone does it",
        "you'll like it",
        "don't be such a baby",
        "don't be such a kid",
        "you said you trusted me",

        # ------------------------------------------------------------------
        # High-risk combinations
        # ------------------------------------------------------------------
        "how old are you? add me on snap and don't tell your parents",
        "are you alone? turn on your camera",
        "send me a selfie and keep this between us",
        "i'll give you robux if you send me a pic",
        "what school do you go to? let's meet after class",
        "don't screenshot this and send me something hot",
        "come alone and don't tell anyone where you're going",
        "i saved your photos, send more or i'll post them",
        "you're mature for your age, let's talk somewhere private",
        "lock your door and video call me alone",

        # ------------------------------------------------------------------
        # False-positive pressure tests
        # ------------------------------------------------------------------
        "send me a picture of the broken laptop",
        "send me your number so i can add you to the school project group",
        "what school do you go to? i am updating the class database",
        "are your parents home? the delivery needs an adult signature",
        "turn on your camera for the online class",
        "delete the messages from the spam bot",
        "clear your browser history if the website is broken",
        "join my private server for the tournament",
        "what time do you leave school for the official trip",
        "don't tell your parents about the surprise party until tomorrow",
    ]

    for text in test_cases:
        result = engine.summary(text)
        sev    = result["max_severity"]
        cats   = ", ".join(sorted(result["categories"])) or "none"
        print(f"[sev={sev}] [{cats}]")
        print(f"  '{text}'")
        for m in result["matches"]:
            print(f"    -> {m.category} sev={m.severity}  matched='{m.text}'")
        print()
