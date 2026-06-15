"""
aml_guard_NO_CLI.py
Single-file AML grooming risk detector and PAN2012 preprocessor.
NO CLI VERSION: run only `python aml_guard_NO_CLI.py`.

Features:
- regex rule engine
- leet-speech normalization
- optional thefuzz typo matching with difflib fallback
- category-aware risk scoring
- PAN2012 train/val/test JSON export

Install for fuzzy speed/quality:
    pip install "thefuzz[speedup]"

Run:
    python aml_guard_NO_CLI.py

The script uses the PAN2012 default paths configured near the bottom of this file
and writes pan12_dataset/train.json, val.json, and test.json.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

try:
    from thefuzz import fuzz as _fuzz
except ImportError:  # graceful fallback; install thefuzz for better matching
    _fuzz = None

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


WINDOW_SIZE = 40
THETA_H = 0.75
THETA_M = 0.50
NH = 2
NM = 3
SCORE_HIGH = 5.0
SCORE_MEDIUM = 1.8

ESCAL_BONUS = 4.0
SEV_JUMP_BONUS = 2.0
STAGE_ARC_BONUS = 3.5
CONTROL_ARC_EXTRA = 1.5
CONSEC_BONUS = 1.0
DOMINANCE_BONUS = 2.0
DENSITY_BONUS = 2.5

FUZZY_MIN_SHORT = 92
FUZZY_MIN = 88
FUZZY_MIN_CRITICAL = 88

SLANG_MAP = {'\\bu\\b': 'you',
 '\\bur\\b': 'your',
 '\\br\\b': 'are',
 '\\by\\b': 'why',
 '\\bpls\\b': 'please',
 '\\bplz\\b': 'please',
 '\\bthx\\b': 'thanks',
 '\\bthanx\\b': 'thanks',
 '\\bty\\b': 'thank you',
 '\\bnp\\b': 'no problem',
 '\\bsry\\b': 'sorry',
 '\\bsrry\\b': 'sorry',
 '\\bidk\\b': 'i do not know',
 '\\bikr\\b': 'i know right',
 '\\bimo\\b': 'in my opinion',
 '\\bimho\\b': 'in my humble opinion',
 '\\bngl\\b': 'not gonna lie',
 '\\btbh\\b': 'to be honest',
 '\\bnvm\\b': 'never mind',
 '\\bsmh\\b': 'shaking my head',
 '\\bfyi\\b': 'for your information',
 '\\blol\\b': 'laughing out loud',
 '\\blmao\\b': 'laughing my ass off',
 '\\brofl\\b': 'rolling on the floor laughing',
 '\\bomg\\b': 'oh my god',
 '\\bomw\\b': 'on my way',
 '\\bbrb\\b': 'be right back',
 '\\bgtg\\b': 'got to go',
 '\\bg2g\\b': 'got to go',
 '\\bttyl\\b': 'talk to you later',
 '\\bcya\\b': 'see you',
 '\\bcu\\b': 'see you',
 '\\bwb\\b': 'welcome back',
 '\\bafk\\b': 'away from keyboard',
 '\\bbtw\\b': 'by the way',
 '\\bafaik\\b': 'as far as i know',
 '\\basap\\b': 'as soon as possible',
 '\\birl\\b': 'in real life',
 '\\bidc\\b': 'i do not care',
 '\\bbf\\b': 'boyfriend',
 '\\bgf\\b': 'girlfriend',
 '\\bbff\\b': 'best friend forever',
 '\\bcrush\\b': 'romantic interest',
 '\\bima\\b': 'i am going to',
 '\\bimma\\b': 'i am going to',
 '\\bgonna\\b': 'going to',
 '\\bwanna\\b': 'want to',
 '\\bgotta\\b': 'got to',
 '\\blemme\\b': 'let me',
 '\\bgimme\\b': 'give me',
 '\\bkinda\\b': 'kind of',
 '\\bsorta\\b': 'sort of',
 '\\boutta\\b': 'out of',
 '\\bcuz\\b': 'because',
 '\\bcoz\\b': 'because',
 '\\bcause\\b': 'because',
 '\\bwyd\\b': 'what are you doing',
 '\\bhmu\\b': 'hit me up',
 '\\bwbu\\b': 'what about you',
 '\\bhbu\\b': 'how about you',
 '\\bwydrn\\b': 'what are you doing right now',
 '\\brn\\b': 'right now',
 '\\btmr\\b': 'tomorrow',
 '\\btmrw\\b': 'tomorrow',
 '\\btonite\\b': 'tonight',
 '\\bl8r\\b': 'later',
 '\\bgr8\\b': 'great',
 '\\bm8\\b': 'mate',
 '\\basl\\b': 'age sex location',
 '\\bpic\\b': 'picture',
 '\\bpics\\b': 'pictures',
 '\\bvid\\b': 'video',
 '\\bvids\\b': 'videos',
 '\\bdm\\b': 'direct message',
 '\\bdms\\b': 'direct messages',
 '\\bpm\\b': 'private message',
 '\\bpms\\b': 'private messages',
 '\\bfr\\b': 'for real',
 '\\bfrfr\\b': 'for real for real',
 '\\bnoob\\b': 'newbie',
 '\\bn00b\\b': 'newbie',
 '\\bppl\\b': 'people',
 '\\bpeeps\\b': 'people',
 '\\bsup\\b': 'what is up',
 '\\bwat\\b': 'what',
 '\\bwut\\b': 'what',
 '\\btho\\b': 'though',
 '\\bthru\\b': 'through',
 '\\bkk\\b': 'okay',
 '\\bokay\\b': 'okay',
 '\\bok\\b': 'okay',
 '\\bk\\b': 'okay',
 '\\byeah\\b': 'yes',
 '\\bye\\b': 'yes',
 '\\byep\\b': 'yes',
 '\\byup\\b': 'yes',
 '\\bnah\\b': 'no',
 '\\bnope\\b': 'no',
 '\\bily\\b': 'i love you',
 '\\bilysm\\b': 'i love you so much',
 '\\bilu\\b': 'i love you',
 '\\bxoxo\\b': 'hugs and kisses',
 '\\bjk\\b': 'just kidding',
 '\\bjkng\\b': 'just kidding',
 '\\bjw\\b': 'just wondering',
 '\\bn1\\b': 'nice one',
 '\\bgg\\b': 'good game',
 '\\bwp\\b': 'well played',
 '\\bwht\\b': 'what',
 '\\bwhts\\b': 'what is',
 '\\bwhats\\b': 'what is',
 '\\bdnt\\b': 'do not',
 '\\bdont\\b': 'do not',
 '\\bnt\\b': 'not',
 '\\bpix\\b': 'picture',
 '\\bpik\\b': 'picture',
 '\\bsnap\\b': 'snapchat',
 '\\bsc\\b': 'snapchat',
 '\\big\\b': 'instagram',
 '\\bdc\\b': 'discord',
 '\\bdisc\\b': 'discord'}

RULES = {'isolation': [(3, "don'?t\\s+tell\\s+(your\\s+)?(parents?|mum|mom|dad|family|friends?)"),
               (3, 'keep\\s+(this|it|our\\s+\\w+)\\s+(a\\s+)?secret'),
               (3, '(just\\s+)between\\s+(you\\s+and\\s+me|us)'),
               (2, "don'?t\\s+(let|show)\\s+\\w+\\s+(see|know|find out)"),
               (2, '(delete|clear)\\s+(this|these|the\\s+)?(messages?|chats?|conv\\w*)'),
               (3, '\\bdo\\s+not\\s+tell\\s+(your\\s+)?(parents?|mum|mom|dad|family|friends?)\\b'),
               (2, '\\barchive\\s+this\\s+chat\\b')],
 'meeting': [(3, "(let'?s|wanna|want\\s+to|wna)\\s+meet(\\s+up)?"),
             (3, 'are\\s+you\\s+(home\\s+)?alone'),
             (3, '(come\\s+to|visit)\\s+(my|our)\\s+(place|house|flat|home|apartment)'),
             (2, '(hang\\s+out|chill)\\s+(with\\s+me|together|sometime)'),
             (2, 'where\\s+do\\s+you\\s+live')],
 'age_probing': [(2, 'how\\s+old\\s+are\\s+you'),
                 (2, "what'?s?\\s+your\\s+age"),
                 (1, "you('?re|\\s+are)\\s+(so\\s+)?(mature|grown[\\s-]up|advanced)\\s+for\\s+your\\s+age"),
                 (1, 'you\\s+(seem|look|act)\\s+(older|more\\s+mature)\\s+than\\s+\\d+')],
 'image_solicitation': [(3, 'send\\s+(me\\s+)?(a\\s+)?(pic(ture)?s?|photo|selfie|snap|nude)'),
                        (3, 'show\\s+me\\s+(your(self)?|a\\s+pic|what\\s+you\\s+look)'),
                        (3, '(got\\s+any\\s+|have\\s+any\\s+)?(nudes?|pics?\\s+of\\s+you)'),
                        (2, '(can\\s+i\\s+see|let\\s+me\\s+see)\\s+(you|your\\s+\\w+)'),
                        (3, '\\bsend\\s+(me\\s+)?(pictures?|photos?|selfies?|snaps?)\\s+of\\s+you\\b'),
                        (3, '\\bsend\\s+(it|one)\\s+to\\s+me\\b')],
 'rapid_intimacy': [(2, "i'?ve?\\s+never\\s+(felt|met|found)\\s+(anyone|someone|a\\s+\\w+)\\s+like\\s+you"),
                    (2, "you'?re?\\s+(the\\s+)?(only\\s+one|most\\s+\\w+\\s+person)\\s+i\\s+(know|trust|have)"),
                    (1, '(i\\s+)?(really\\s+)?(like|love|adore)\\s+you\\s+(so\\s+much|a\\s+lot)'),
                    (1, 'you\\s+(are|r)\\s+(so\\s+)?(beautiful|gorgeous|cute|hot|sexy|perfect)')],
 'coercion': [(3, "i'?ll\\s+(tell|show|send)\\s+(everyone|your\\s+(parents?|friends?|school))"),
              (2, 'you\\s+(promised|said\\s+you\\s+would)'),
              (2, 'i\\s+thought\\s+you\\s+(trusted?|liked?|loved?)\\s+me'),
              (2, "(after\\s+everything|after\\s+all)\\s+i'?ve?\\s+(done|given)\\s+(for\\s+you|you)")],
 'sexual_content': [(3, '\\b(sex|sexual|intercourse|masturbat\\w+|orgasm|erect\\w+)\\b'),
                    (3, '\\b(cock|dick|pussy|vagina|penis|boobs?|breasts?|naked|nude)\\b')],
 'platform_migration': [(3,
                         '\\b(add|dm|message|text)\\s+me\\s+on\\s+(snap(chat)?|telegram|whatsapp|kik|discord|insta|instagram)\\b'),
                        (3, "\\bwhat'?s\\s+your\\s+(snap(chat)?|telegram|whatsapp|kik|discord|insta|instagram)\\b"),
                        (3, "\\b(let'?s|we\\s+should)\\s+(talk|chat|message)\\s+(somewhere\\s+)?private\\b"),
                        (2, '\\buse\\s+(vanish\\s+mode|disappearing\\s+messages|secret\\s+chat)\\b'),
                        (2, '\\bturn\\s+on\\s+(vanish\\s+mode|disappearing\\s+messages)\\b'),
                        (2, "\\bdon'?t\\s+(save|screenshot|screen\\s*record)\\s+(this|it|our\\s+chat|messages?)\\b")],
 'supervision_probe': [(3, '\\bare\\s+your\\s+(parents?|mum|mom|dad|family)\\s+(home|there|around)\\b'),
                       (3, '\\bare\\s+you\\s+alone\\s+(right\\s+now|rn|at\\s+home)?\\b'),
                       (3, '\\bis\\s+anyone\\s+(watching|with\\s+you|near\\s+you|in\\s+the\\s+room)\\b'),
                       (2, '\\bcan\\s+anyone\\s+(see|read|check)\\s+your\\s+(phone|screen|messages?|chat)\\b'),
                       (2,
                        '\\bdo\\s+your\\s+parents?\\s+(check|read|monitor)\\s+your\\s+(phone|messages?|apps?|account)\\b'),
                       (2, '\\b(lock|close)\\s+your\\s+(door|bedroom\\s+door)\\b')],
 'contact_info_probe': [(3,
                         '\\b(send|give|share)\\s+(me\\s+)?your\\s+(number|phone\\s+number|address|location|pin)\\b'),
                        (3, "\\bwhat'?s\\s+your\\s+(number|phone\\s+number|address|location)\\b"),
                        (2, '\\bdrop\\s+your\\s+(pin|location|snap|insta|discord)\\b'),
                        (2, '\\bwhere\\s+exactly\\s+do\\s+you\\s+live\\b'),
                        (2, "\\bwhat'?s\\s+your\\s+(full\\s+name|last\\s+name|surname)\\b"),
                        (2, '\\bwhat\\s+school\\s+do\\s+you\\s+go\\s+to\\b'),
                        (2, '\\bwhat\\s+(bus|train|route)\\s+do\\s+you\\s+take\\b')],
 'routine_probe': [(2, '\\bwhat\\s+time\\s+do\\s+you\\s+(leave|finish|get\\s+out\\s+of)\\s+(school|class|practice)\\b'),
                   (2, '\\bwhen\\s+are\\s+you\\s+(home|alone|free)\\b'),
                   (2, '\\bwhat\\s+days\\s+are\\s+your\\s+parents?\\s+(away|working|not\\s+home)\\b'),
                   (2, '\\bwhen\\s+do\\s+you\\s+walk\\s+home\\b'),
                   (2, '\\bwhere\\s+do\\s+you\\s+hang\\s+out\\s+after\\s+school\\b')],
 'gifts_incentives': [(3,
                       "\\bi('?ll| "
                       'will)\\s+(buy|get|send|give)\\s+you\\s+(a\\s+)?(gift|present|phone|card|gift\\s*card)\\b'),
                      (3, "\\bi('?ll| will)\\s+pay\\s+you\\b"),
                      (3, '\\b(send|give)\\s+you\\s+(money|cash|paypal|crypto|gift\\s*cards?)\\b'),
                      (2, '\\b(robux|v-?bucks|nitro|steam\\s+card|xbox\\s+card|playstation\\s+card)\\b'),
                      (2, "\\bi('?ll| will)\\s+(pick\\s+you\\s+up|drive\\s+you|get\\s+you\\s+a\\s+ride)\\b"),
                      (2, "\\bi('?ll| will)\\s+get\\s+you\\s+(weed|alcohol|vape|cigarettes?)\\b")],
 'boundary_testing': [(2, '\\bhave\\s+you\\s+ever\\s+(kissed|dated|made\\s+out)\\b'),
                      (2, '\\bhave\\s+you\\s+had\\s+your\\s+first\\s+(kiss|boyfriend|girlfriend)\\b'),
                      (2, '\\bare\\s+you\\s+(a\\s+)?virgin\\b'),
                      (2, "\\bwhat'?s\\s+your\\s+body\\s+count\\b"),
                      (2, '\\bdo\\s+you\\s+like\\s+(older\\s+guys|older\\s+girls|older\\s+men|older\\s+women)\\b'),
                      (2, '\\bwould\\s+you\\s+date\\s+someone\\s+older\\b'),
                      (1, "\\byou'?re\\s+not\\s+(a\\s+)?little\\s+(kid|child)\\s+anymore\\b")],
 'sexual_escalation': [(3, '\\b(send|show)\\s+(me\\s+)?something\\s+(sexy|hot|spicy)\\b'),
                       (3, '\\bmake\\s+me\\s+(hard|wet)\\b'),
                       (3, '\\bturn\\s+me\\s+on\\b'),
                       (3, "\\blet'?s\\s+(sext|trade\\s+nudes)\\b"),
                       (2, "\\blet'?s\\s+(roleplay|rp)\\b"),
                       (2, '\\bdirty\\s+talk\\b'),
                       (2, '\\bwhat\\s+are\\s+you\\s+wearing\\b'),
                       (2, '\\brate\\s+my\\s+(body|pic|photo)\\b'),
                       (3, '\\bsend\\s+(a\\s+)?thirst\\s+trap\\b'),
                       (2, '\\bshow\\s+a\\s+little\\s+more\\b')],
 'body_focus': [(3, '\\bshow\\s+me\\s+your\\s+(body|chest|butt|ass|legs|stomach|tummy)\\b'),
                (3, '\\bsend\\s+(me\\s+)?(a\\s+)?(body\\s+pic|full\\s+body\\s+pic|mirror\\s+selfie)\\b'),
                (2, '\\bstand\\s+up\\s+so\\s+i\\s+can\\s+see\\s+you\\b'),
                (2, '\\bturn\\s+around\\s+for\\s+me\\b'),
                (2, '\\blift\\s+your\\s+(shirt|top)\\b'),
                (2, '\\bshow\\s+me\\s+your\\s+outfit\\b')],
 'live_video_pressure': [(3, '\\bturn\\s+on\\s+your\\s+(camera|cam|webcam)\\b'),
                         (3, '\\bvideo\\s+call\\s+me\\s+(alone|privately|now)\\b'),
                         (2, "\\blet'?s\\s+(facetime|video\\s+chat|cam)\\b"),
                         (2, '\\bgo\\s+live\\s+for\\s+me\\b'),
                         (2, '\\bcan\\s+i\\s+watch\\s+you\\b'),
                         (2, '\\bmove\\s+the\\s+camera\\s+(down|lower|closer)\\b')],
 'reciprocal_image_pressure': [(3,
                                "\\bi('?ll| will)\\s+send\\s+(one|mine|first)\\s+if\\s+you\\s+send\\s+(one|yours)\\b"),
                               (3, '\\byour\\s+turn\\s+(now\\s+)?to\\s+send\\b'),
                               (3, '\\bi\\s+showed\\s+you\\s+mine\\b'),
                               (3, '\\bprove\\s+(it|you\\s+trust\\s+me)\\s+with\\s+(a\\s+)?(pic|photo|snap|selfie)\\b'),
                               (2, '\\bjust\\s+one\\s+(pic|photo|snap|selfie)\\b'),
                               (2, "\\bi\\s+promise\\s+i('?ll| will)\\s+delete\\s+it\\b")],
 'sextortion': [(3, '\\bi\\s+(have|saved|recorded|screenshotted)\\s+(your\\s+)?(pics?|photos?|videos?|nudes?)\\b'),
                (3, "\\bi('?ll| will)\\s+(post|leak|share|send)\\s+(them|it|your\\s+\\w+)\\b"),
                (3, "\\b(send\\s+more|do\\s+what\\s+i\\s+say)\\s+or\\s+i('?ll| will)\\b"),
                (3, '\\bi\\s+know\\s+your\\s+(school|parents?|friends?|address)\\b'),
                (3, "\\bpay\\s+me\\s+or\\s+i('?ll| will)\\s+(post|leak|send|share)\\b"),
                (3, "\\bdon'?t\\s+block\\s+me\\s+or\\s+i('?ll| will)\\b")],
 'age_gap_minimization': [(2, '\\bage\\s+(is\\s+)?just\\s+a\\s+number\\b'),
                          (2, "\\bdon'?t\\s+worry\\s+about\\s+my\\s+age\\b"),
                          (2, "\\byou'?re\\s+basically\\s+an\\s+adult\\b"),
                          (2, "\\byou'?re\\s+mature\\s+enough\\b"),
                          (2, '\\bnobody\\s+has\\s+to\\s+know\\s+how\\s+old\\s+we\\s+are\\b'),
                          (2, '\\bpeople\\s+our\\s+age\\s+do\\s+this\\b')],
 'dependency_building': [(2, '\\bi\\s+understand\\s+you\\s+better\\s+than\\s+(your\\s+)?(parents?|family|friends?)\\b'),
                         (2, "\\bthey\\s+don'?t\\s+understand\\s+you\\s+like\\s+i\\s+do\\b"),
                         (2, "\\bi('?m| am)\\s+the\\s+only\\s+one\\s+who\\s+(cares|gets\\s+you|understands)\\b"),
                         (2, '\\byou\\s+need\\s+me\\b'),
                         (2, "\\byou\\s+can'?t\\s+trust\\s+(your\\s+)?(parents?|family|friends?)\\b"),
                         (2, '\\bwe\\s+have\\s+a\\s+special\\s+connection\\b')],
 'offline_evasion': [(3, '\\bsneak\\s+out\\b'),
                     (3, "\\bdon'?t\\s+tell\\s+anyone\\s+where\\s+you'?re\\s+going\\b"),
                     (3, '\\bcome\\s+alone\\b'),
                     (3, '\\bbring\\s+(no\\s+one|nobody)\\b'),
                     (3, "\\bi('?ll| will)\\s+pick\\s+you\\s+up\\s+(after\\s+school|tonight|tomorrow)\\b"),
                     (2, '\\bmeet\\s+me\\s+(behind|near|outside)\\s+(the\\s+)?(school|mall|park|station)\\b'),
                     (2, '\\buse\\s+the\\s+back\\s+door\\b')],
 'account_evasion': [(3, '\\bmake\\s+(a\\s+)?(secret|private|new|alt|burner)\\s+account\\b'),
                     (3, "\\bdon'?t\\s+use\\s+your\\s+main\\s+account\\b"),
                     (2, '\\bhide\\s+(this|our\\s+chat|the\\s+app)\\b'),
                     (2, '\\bchange\\s+my\\s+name\\s+in\\s+your\\s+phone\\b'),
                     (2, '\\bdelete\\s+my\\s+contact\\b'),
                     (2, '\\bclear\\s+your\\s+(history|notifications|recent\\s+apps)\\b'),
                     (2, '\\barchive\\s+this\\s+chat\\b')],
 'identity_deception': [(2, "\\bi('?m| am)\\s+(your\\s+age|the\\s+same\\s+age)\\b"),
                        (2, "\\bi('?m| am)\\s+\\d{1,2}\\s+too\\b"),
                        (2, '\\bi\\s+look\\s+younger\\s+than\\s+i\\s+am\\b'),
                        (2, "\\bdon'?t\\s+ask\\s+too\\s+many\\s+questions\\s+about\\s+me\\b"),
                        (2, "\\bi\\s+can'?t\\s+show\\s+my\\s+face\\b"),
                        (1, '\\bmy\\s+camera\\s+is\\s+broken\\b')],
 'jealousy_control': [(2, "\\bdon'?t\\s+talk\\s+to\\s+other\\s+(guys|girls|boys|men|women)\\b"),
                      (2, '\\bwho\\s+else\\s+are\\s+you\\s+talking\\s+to\\b'),
                      (2, '\\bwhy\\s+were\\s+you\\s+online\\s+and\\s+not\\s+answering\\s+me\\b'),
                      (3, '\\byou\\s+belong\\s+to\\s+me\\b'),
                      (2, '\\bprove\\s+you\\s+(love|trust)\\s+me\\b')],
 'persistence_tracking': [(1, "\\bwhy\\s+aren'?t\\s+you\\s+answering\\s+me\\b"),
                          (1, '\\bi\\s+saw\\s+you\\s+were\\s+online\\b'),
                          (2, '\\bi\\s+found\\s+your\\s+other\\s+account\\b'),
                          (2, '\\bi\\s+found\\s+your\\s+(insta|instagram|snap|snapchat|discord)\\b'),
                          (1, '\\bwhy\\s+did\\s+you\\s+block\\s+me\\b'),
                          (2, '\\banswer\\s+me\\s+(right\\s+now|rn)\\b'),
                          (1, '\\bi\\s+know\\s+you\\s+read\\s+this\\b')],
 'gaming_private_context': [(2, '\\bjoin\\s+my\\s+private\\s+server\\b'),
                            (2, '\\bcome\\s+to\\s+my\\s+private\\s+lobby\\b'),
                            (2, "\\bi('?ll| will)\\s+gift\\s+you\\s+(skins|robux|v-?bucks|nitro)\\b"),
                            (2, '\\buse\\s+voice\\s+chat\\s+with\\s+me\\s+alone\\b'),
                            (2, "\\bdon'?t\\s+invite\\s+your\\s+friends\\b"),
                            (2, '\\bparty\\s+chat\\s+only\\b')],
 'late_night_contact': [(2, '\\b(text|message|call)\\s+me\\s+when\\s+everyone\\s+is\\s+asleep\\b'),
                        (2, '\\b(message|text|call)\\s+me\\s+after\\s+midnight\\b'),
                        (1, '\\bstay\\s+up\\s+with\\s+me\\s+tonight\\b'),
                        (1, "\\bdon'?t\\s+fall\\s+asleep\\s+yet\\b")],
 'consent_minimization': [(2, '\\bstop\\s+being\\s+(shy|scared)\\s+and\\s+send\\s+it\\b'),
                          (2, "\\bit'?s\\s+not\\s+a\\s+big\\s+deal\\s+just\\s+send\\s+it\\b"),
                          (2, '\\beveryone\\s+does\\s+it\\b'),
                          (2, "\\byou('?ll| will)\\s+like\\s+it\\b"),
                          (2, "\\bdon'?t\\s+be\\s+such\\s+a\\s+(baby|kid)\\b"),
                          (2, '\\byou\\s+said\\s+you\\s+trusted\\s+me\\b')]}

CAT_WEIGHT = {'sextortion': 5.0,
 'sexual_content': 4.5,
 'sexual_escalation': 4.5,
 'image_solicitation': 4.0,
 'body_focus': 4.0,
 'reciprocal_image_pressure': 4.0,
 'live_video_pressure': 3.6,
 'isolation': 3.5,
 'offline_evasion': 3.5,
 'coercion': 3.5,
 'supervision_probe': 2.8,
 'meeting': 2.8,
 'platform_migration': 2.6,
 'contact_info_probe': 2.6,
 'gifts_incentives': 2.4,
 'consent_minimization': 2.4,
 'account_evasion': 2.3,
 'jealousy_control': 2.2,
 'identity_deception': 2.0,
 'late_night_contact': 2.0,
 'dependency_building': 1.9,
 'age_gap_minimization': 1.9,
 'boundary_testing': 1.8,
 'routine_probe': 1.8,
 'gaming_private_context': 1.8,
 'persistence_tracking': 1.6,
 'rapid_intimacy': 1.6,
 'age_probing': 1.0}

COOCCUR_BONUS = [({'sexual_content', 'coercion'}, 3.0),
 ({'sexual_escalation', 'coercion'}, 3.0),
 ({'image_solicitation', 'coercion'}, 2.5),
 ({'sextortion', 'image_solicitation'}, 3.0),
 ({'sextortion', 'coercion'}, 3.0),
 ({'sexual_content', 'isolation'}, 2.5),
 ({'isolation', 'sexual_escalation'}, 2.5),
 ({'image_solicitation', 'isolation'}, 2.5),
 ({'supervision_probe', 'meeting'}, 2.2),
 ({'offline_evasion', 'meeting'}, 3.0),
 ({'meeting', 'contact_info_probe'}, 2.2),
 ({'image_solicitation', 'gifts_incentives'}, 2.5),
 ({'sexual_content', 'platform_migration'}, 2.2),
 ({'image_solicitation', 'platform_migration'}, 2.2),
 ({'dependency_building', 'image_solicitation'}, 2.2),
 ({'dependency_building', 'isolation'}, 2.0),
 ({'sexual_escalation', 'boundary_testing'}, 2.8),
 ({'image_solicitation', 'boundary_testing'}, 2.2),
 ({'image_solicitation', 'rapid_intimacy'}, 2.2),
 ({'sexual_content', 'rapid_intimacy'}, 2.8),
 ({'supervision_probe', 'gifts_incentives'}, 2.0),
 ({'image_solicitation', 'account_evasion'}, 2.2),
 ({'sexual_content', 'account_evasion'}, 2.2),
 ({'supervision_probe', 'late_night_contact'}, 2.0),
 ({'image_solicitation', 'late_night_contact'}, 2.3),
 ({'gifts_incentives', 'gaming_private_context'}, 2.0),
 ({'platform_migration', 'gaming_private_context'}, 2.0),
 ({'jealousy_control', 'coercion'}, 2.3),
 ({'persistence_tracking', 'coercion'}, 2.0),
 ({'image_solicitation', 'consent_minimization'}, 2.5),
 ({'identity_deception', 'platform_migration'}, 2.0)]

TRUST_CATS = {'age_gap_minimization',
 'boundary_testing',
 'dependency_building',
 'gifts_incentives',
 'identity_deception',
 'rapid_intimacy'}
SOLICIT_CATS = {'body_focus',
 'consent_minimization',
 'image_solicitation',
 'live_video_pressure',
 'reciprocal_image_pressure',
 'sextortion',
 'sexual_content',
 'sexual_escalation'}
CONTROL_CATS = {'account_evasion',
 'coercion',
 'isolation',
 'jealousy_control',
 'late_night_contact',
 'offline_evasion',
 'persistence_tracking',
 'platform_migration',
 'supervision_probe'}
FUZZY_PHRASES = {'isolation': [(3, 'do not tell your parents'),
               (3, 'dont tell your parents'),
               (3, 'keep this a secret'),
               (3, 'just between you and me'),
               (2, 'delete these messages'),
               (2, 'clear this chat')],
 'meeting': [(3, 'lets meet up'),
             (3, 'are you home alone'),
             (3, 'come to my place'),
             (2, 'where do you live')],
 'image_solicitation': [(3, 'send me a picture'),
                        (3, 'send me a photo'),
                        (3, 'send me a selfie'),
                        (3, 'send me a nude'),
                        (3, 'show me yourself'),
                        (3, 'got any pictures of you'),
                        (2, 'let me see you')],
 'sexual_escalation': [(3, 'send me something sexy'),
                       (3, 'show me something hot'),
                       (3, 'turn me on'),
                       (3, 'lets trade nudes'),
                       (2, 'what are you wearing'),
                       (2, 'dirty talk')],
 'sexual_content': [(3, 'talk about sex'), (3, 'send a nude'), (3, 'are you naked')],
 'sextortion': [(3, 'i have your pictures'),
                (3, 'i saved your photos'),
                (3, 'i will leak it'),
                (3, 'send more or i will post'),
                (3, 'do what i say or i will'),
                (3, 'pay me or i will leak')],
 'platform_migration': [(3, 'add me on snapchat'),
                        (3, 'message me on telegram'),
                        (3, 'what is your discord'),
                        (3, 'lets talk somewhere private'),
                        (2, 'use vanish mode'),
                        (2, 'do not screenshot this')],
 'supervision_probe': [(3, 'are your parents home'),
                       (3, 'are you alone right now'),
                       (3, 'is anyone watching'),
                       (2, 'lock your bedroom door')],
 'contact_info_probe': [(3, 'send me your number'),
                        (3, 'send me your location'),
                        (3, 'what is your address'),
                        (2, 'what school do you go to')],
 'gifts_incentives': [(3, 'i will pay you'),
                      (3, 'i will buy you a gift'),
                      (3, 'send you money'),
                      (2, 'i will give you robux')],
 'body_focus': [(3, 'show me your body'),
                (3, 'send me a body picture'),
                (2, 'lift your shirt'),
                (2, 'turn around for me')],
 'live_video_pressure': [(3, 'turn on your camera'),
                         (3, 'video call me alone'),
                         (2, 'move the camera down')],
 'reciprocal_image_pressure': [(3, 'i will send mine if you send yours'),
                               (3, 'your turn to send'),
                               (3, 'prove you trust me with a picture'),
                               (2, 'just one picture')],
 'offline_evasion': [(3, 'sneak out'),
                     (3, 'come alone'),
                     (3, 'do not tell anyone where you are going'),
                     (2, 'meet me behind the school')],
 'account_evasion': [(3, 'make a secret account'),
                     (3, 'do not use your main account'),
                     (2, 'hide this chat'),
                     (2, 'clear your history')],
 'consent_minimization': [(2, 'stop being shy and send it'),
                          (2, 'it is not a big deal just send it'),
                          (2, 'everyone does it')],
 'jealousy_control': [(3, 'you belong to me'),
                      (2, 'prove you love me'),
                      (2, 'who else are you talking to')],
 'late_night_contact': [(2, 'message me after midnight'), (2, 'text me when everyone is asleep')]}
BENIGN_EXCEPTIONS = {'image_solicitation': ['\\bsend\\s+me\\s+(a\\s+)?(picture|photo|pic)\\s+of\\s+(your\\s+|the\\s+)?(homework|notes|class\\s+notes|broken\\s+laptop|screen|error|receipt|document)\\b',
                        '\\bsend\\s+me\\s+(the\\s+)?(homework|notes|class\\s+notes|document|file)\\b'],
 'meeting': ["\\blet'?s\\s+meet\\s+at\\s+school\\s+with\\s+(your\\s+)?(teacher|parent|parents|class)\\b"],
 'age_probing': ['\\bhow\\s+old\\s+are\\s+you\\s+.*\\b(users?|server|account)\\s+(over|under)\\s+\\d+\\b'],
 'supervision_probe': ['\\bare\\s+your\\s+parents\\s+home\\s+.*\\b(delivery|signature|permission)\\b'],
 'contact_info_probe': ['\\bsend\\s+me\\s+your\\s+(number|phone\\s+number)\\s+so\\s+i\\s+can\\s+add\\s+you\\s+to\\s+(the\\s+)?(school\\s+)?project\\s+group\\b',
                        '\\bwhat\\s+school\\s+do\\s+you\\s+go\\s+to\\s+.*\\b(class\\s+database|official|registration)\\b'],
 'routine_probe': ['\\bwhat\\s+time\\s+do\\s+you\\s+leave\\s+school\\s+for\\s+(the\\s+)?official\\s+trip\\b'],
 'gaming_private_context': ['\\bjoin\\s+my\\s+private\\s+server\\s+for\\s+(the\\s+)?tournament\\b'],
 'live_video_pressure': ['\\bturn\\s+on\\s+your\\s+camera\\s+for\\s+(the\\s+)?(online\\s+class|class|lesson|meeting)\\b'],
 'isolation': ["\\bdon'?t\\s+tell\\s+your\\s+parents\\s+about\\s+the\\s+surprise\\s+party\\b",
               '\\bdelete\\s+the\\s+messages\\s+from\\s+the\\s+spam\\s+bot\\b']}


SLANG = [(re.compile(p, re.IGNORECASE), v) for p, v in SLANG_MAP.items()]
COMPILED = {
    cat: [(sev, raw, re.compile(raw, re.IGNORECASE)) for sev, raw in patterns]
    for cat, patterns in RULES.items()
}
BENIGN_COMPILED = {
    cat: [re.compile(raw, re.IGNORECASE) for raw in patterns]
    for cat, patterns in BENIGN_EXCEPTIONS.items()
}
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b\d[\d\s\-]{5,13}\d\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")
WS_RE = re.compile(r"\s+")
SPACED_WORD_RE = re.compile(r"(?<![a-z])(?:[a-z]\s+){2,}[a-z](?![a-z])")
JOINER_RE = re.compile(r"(?<=[a-z0-9])[._*`~](?=[a-z0-9])")

LEET_TRANS = str.maketrans({
    "0": "o",
    "1": "i",
    "2": "z",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "+": "t",
    "!": "i",
    "|": "i",
})


def _collapse_spaced_word(match: re.Match) -> str:
    return match.group(0).replace(" ", "")


def deobfuscate_leet(text: str) -> str:
    text = JOINER_RE.sub("", text)
    text = text.translate(LEET_TRANS)
    text = SPACED_WORD_RE.sub(_collapse_spaced_word, text)
    return text


def normalise_phrase(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = deobfuscate_leet(text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = re.sub(r"[^a-z0-9<> ]+", " ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = deobfuscate_leet(text)
    text = REPEAT_RE.sub(r"\1\1", text)
    for pat, rep in SLANG:
        text = pat.sub(rep, text)
    text = re.sub(r"[^a-z0-9<>@'. -]+", " ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


normalize = normalise

FUZZY_CANON = {
    cat: [(sev, normalise_phrase(phrase)) for sev, phrase in phrases]
    for cat, phrases in FUZZY_PHRASES.items()
}


def _ratio(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if _fuzz is not None:
        return max(_fuzz.ratio(a, b), _fuzz.partial_ratio(a, b), _fuzz.token_set_ratio(a, b))
    return int(100 * SequenceMatcher(None, a, b).ratio())


def _ngrams(tokens: list[str], min_n: int, max_n: int):
    n_tokens = len(tokens)
    for n in range(min_n, min(max_n, n_tokens) + 1):
        for i in range(0, n_tokens - n + 1):
            yield i, i + n, " ".join(tokens[i:i + n])


@dataclass
class RuleMatch:
    category: str
    severity: int
    pattern: str
    span: tuple[int, int]
    text: str
    method: str = "regex"
    score: int = 100


@dataclass
class Message:
    text: str
    sender: str
    ml_prob: float
    rule_matches: list[RuleMatch]
    max_severity: int = 0
    categories: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.rule_matches:
            self.max_severity = max(m.severity for m in self.rule_matches)
            self.categories = {m.category for m in self.rule_matches}


@dataclass
class Alert:
    level: str
    reason: str
    message: str
    ml_prob: float
    rule_matches: list[RuleMatch] = field(default_factory=list)
    score: float = 0.0
    signals: dict = field(default_factory=dict)

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
            lines.append(
                f"  rule     : [{rm.category} sev={rm.severity} {rm.method}={rm.score}] '{rm.text}'"
            )
        return "\n".join(lines)


class RuleEngine:
    def __init__(self, use_fuzzy: bool = True) -> None:
        self.use_fuzzy = use_fuzzy

    def match(self, text: str, already_normalised: bool = False) -> list[RuleMatch]:
        norm = text if already_normalised else normalise(text)
        results: list[RuleMatch] = []

        for category, patterns in COMPILED.items():
            for severity, raw, regex in patterns:
                for m in regex.finditer(norm):
                    results.append(RuleMatch(
                        category=category,
                        severity=severity,
                        pattern=raw,
                        span=m.span(),
                        text=m.group(),
                    ))

        if self.use_fuzzy:
            results.extend(self._fuzzy_match(norm, results))

        results = self._suppress_benign(norm, results)
        return self._dedupe(results)

    def _fuzzy_match(self, norm: str, existing: list[RuleMatch]) -> list[RuleMatch]:
        tokens = [t for t in norm.split() if not t.startswith("<")]
        if len(tokens) < 2:
            return []

        exact_keys = {(m.category, m.severity) for m in existing}
        out: list[RuleMatch] = []
        spans = []
        pos = 0
        for tok in norm.split():
            start = norm.find(tok, pos)
            end = start + len(tok)
            spans.append((start, end))
            pos = end

        for category, phrases in FUZZY_CANON.items():
            for severity, phrase in phrases:
                phrase_tokens = phrase.split()
                if len(phrase_tokens) < 2:
                    continue
                min_n = max(2, len(phrase_tokens) - 1)
                max_n = min(8, len(phrase_tokens) + 2)
                cutoff = FUZZY_MIN_CRITICAL if severity >= 3 else FUZZY_MIN
                if len(phrase) <= 12:
                    cutoff = max(cutoff, FUZZY_MIN_SHORT)

                best_score = 0
                best_span = (0, 0)
                best_text = ""
                for i, j, chunk in _ngrams(tokens, min_n, max_n):
                    score = _ratio(chunk, phrase)
                    if score > best_score:
                        best_score = score
                        best_span = (spans[i][0], spans[j - 1][1])
                        best_text = chunk

                if best_score >= cutoff and (category, severity) not in exact_keys:
                    out.append(RuleMatch(
                        category=category,
                        severity=severity,
                        pattern=phrase,
                        span=best_span,
                        text=best_text,
                        method="fuzzy",
                        score=best_score,
                    ))
        return out

    def _suppress_benign(self, norm: str, matches: list[RuleMatch]) -> list[RuleMatch]:
        if not matches:
            return matches
        suppressed = set()
        for cat, patterns in BENIGN_COMPILED.items():
            if any(p.search(norm) for p in patterns):
                suppressed.add(cat)
        if not suppressed:
            return matches
        high_risk = {m.category for m in matches if m.category not in suppressed and m.severity >= 2}
        if high_risk & SOLICIT_CATS:
            return matches
        return [m for m in matches if m.category not in suppressed]

    def _dedupe(self, matches: list[RuleMatch]) -> list[RuleMatch]:
        best: dict[tuple[str, int, str], RuleMatch] = {}
        for m in matches:
            key = (m.category, m.severity, m.text)
            prev = best.get(key)
            if prev is None or (m.score, len(m.text)) > (prev.score, len(prev.text)):
                best[key] = m
        return sorted(best.values(), key=lambda x: (x.span[0], -x.severity, x.category))

    def max_severity(self, text: str) -> int:
        matches = self.match(text)
        return max((m.severity for m in matches), default=0)

    def has_critical(self, text: str) -> bool:
        return self.max_severity(text) >= 3

    def summary(self, text: str) -> dict:
        matches = self.match(text)
        return {
            "normalised": normalise(text),
            "matches": matches,
            "max_severity": max((m.severity for m in matches), default=0),
            "categories": {m.category for m in matches},
            "critical": any(m.severity == 3 for m in matches),
        }


_RULE_ENGINE = RuleEngine()


def risk_score(cat_counts: dict[str, int], pred_msgs: list[dict], total_msgs: int) -> tuple[float, str, dict]:
    active = {cat for cat, cnt in cat_counts.items() if cnt > 0}
    n_pred = len(pred_msgs)

    base = sum(CAT_WEIGHT.get(cat, 1.0) * min(cnt, 3) for cat, cnt in cat_counts.items())
    cooccur = sum(bonus for pair, bonus in COOCCUR_BONUS if pair.issubset(active))

    escal = 0.0
    if n_pred >= 4:
        half = n_pred // 2
        early_sev = max((m["max_sev"] for m in pred_msgs[:half]), default=0)
        late_sev = max((m["max_sev"] for m in pred_msgs[half:]), default=0)
        if late_sev > early_sev:
            escal = ESCAL_BONUS
            if early_sev <= 1 and late_sev == 3:
                escal += SEV_JUMP_BONUS

    arc = 0.0
    if n_pred >= 5:
        third = max(n_pred // 3, 1)
        early_cats = {cat for m in pred_msgs[:third] for cat in m["cats"]}
        late_cats = {cat for m in pred_msgs[-third:] for cat in m["cats"]}
        if early_cats & TRUST_CATS and late_cats & SOLICIT_CATS:
            arc = STAGE_ARC_BONUS
            if active & CONTROL_CATS:
                arc += CONTROL_ARC_EXTRA

    dominance = 0.0
    if total_msgs > 0 and n_pred >= 4 and n_pred / total_msgs > 0.60:
        dominance = DOMINANCE_BONUS

    total_hits = sum(len(m["matches"]) for m in pred_msgs)
    density = 0.0
    if n_pred > 0 and total_hits / n_pred > 1.25:
        density = DENSITY_BONUS

    consec = 0.0
    run = 0
    for i, m in enumerate(pred_msgs):
        if i == 0:
            run = 1
            continue
        if m["conv_idx"] == pred_msgs[i - 1]["conv_idx"] + 1:
            run += 1
            if run >= 3:
                consec += CONSEC_BONUS
        else:
            run = 1

    score = base + cooccur + escal + arc + dominance + density + consec
    if score >= SCORE_HIGH:
        level = "HIGH"
    elif score >= SCORE_MEDIUM:
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
        "cats": sorted(active),
    }
    return score, level, breakdown


class ContextModel:
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        theta_h: float = THETA_H,
        theta_m: float = THETA_M,
        nh: int = NH,
        nm: int = NM,
        score_high: float = SCORE_HIGH,
        score_medium: float = SCORE_MEDIUM,
    ) -> None:
        self.window_size = window_size
        self.theta_h = theta_h
        self.theta_m = theta_m
        self.nh = nh
        self.nm = nm
        self.score_high = score_high
        self.score_medium = score_medium
        self.window: deque[Message] = deque(maxlen=window_size)

    def reset(self) -> None:
        self.window.clear()

    def _ext(self) -> list[Message]:
        return [m for m in self.window if m.sender == "external"]

    def _high_risk_count(self) -> int:
        return sum(1 for m in self._ext() if m.ml_prob >= self.theta_h)

    def _risky_count(self) -> int:
        return sum(1 for m in self._ext() if m.ml_prob >= self.theta_m or m.max_severity >= 2)

    def _window_risk(self) -> tuple[float, dict]:
        ext = self._ext()
        if not ext:
            return 0.0, {"cats": []}

        cat_counts: dict[str, int] = {}
        pred_msgs: list[dict] = []
        for conv_idx, msg in enumerate(self.window):
            if msg.sender != "external":
                continue
            for cat in msg.categories:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            pred_msgs.append({
                "conv_idx": conv_idx,
                "max_sev": msg.max_severity,
                "cats": msg.categories,
                "matches": msg.rule_matches,
            })
        score, _, bd = risk_score(cat_counts, pred_msgs, len(self.window))
        bd["escal_flag"] = bd.get("escal", 0) > 0
        bd["arc_flag"] = bd.get("arc", 0) > 0
        return score, bd

    def update(self, text: str, sender: str, ml_prob: float, rule_matches: Optional[list[RuleMatch]] = None):
        norm = normalise(text)
        if rule_matches is None:
            rule_matches = _RULE_ENGINE.match(norm, already_normalised=True) if sender == "external" else []
        msg = Message(text=norm, sender=sender, ml_prob=ml_prob, rule_matches=rule_matches)
        self.window.append(msg)

        if sender != "external":
            return None

        score, bd = self._window_risk()
        cats_str = ", ".join(bd.get("cats", [])) or "none"
        signals = {
            "ESC": "yes" if bd.get("escal_flag") else "",
            "ARC": "yes" if bd.get("arc_flag") else "",
        }

        if msg.max_severity == 3:
            return Alert(
                level="Critical",
                reason=f"Severity-3 rule match | cats=[{', '.join(sorted(msg.categories))}]",
                message=norm,
                ml_prob=ml_prob,
                rule_matches=rule_matches,
                score=score,
                signals=signals,
            )

        if self._high_risk_count() >= self.nh:
            return Alert(
                level="High",
                reason=f"High ML count={self._high_risk_count()} >= {self.nh} | score={score:.1f} | cats=[{cats_str}]",
                message=norm,
                ml_prob=ml_prob,
                rule_matches=rule_matches,
                score=score,
                signals=signals,
            )

        if score >= self.score_high:
            return Alert(
                level="High",
                reason=f"Window risk={score:.1f} >= {self.score_high} | cats=[{cats_str}]",
                message=norm,
                ml_prob=ml_prob,
                rule_matches=rule_matches,
                score=score,
                signals=signals,
            )

        if self._risky_count() >= self.nm:
            return Alert(
                level="Medium",
                reason=f"Sustained risk count={self._risky_count()} >= {self.nm} | score={score:.1f} | cats=[{cats_str}]",
                message=norm,
                ml_prob=ml_prob,
                rule_matches=rule_matches,
                score=score,
                signals=signals,
            )

        if score >= self.score_medium:
            return Alert(
                level="Medium",
                reason=f"Window risk={score:.1f} >= {self.score_medium} | cats=[{cats_str}]",
                message=norm,
                ml_prob=ml_prob,
                rule_matches=rule_matches,
                score=score,
                signals=signals,
            )
        return None

    def stats(self) -> dict:
        score, bd = self._window_risk()
        return {
            "window_len": len(self.window),
            "high_risk_count": self._high_risk_count(),
            "risky_count": self._risky_count(),
            "score": round(score, 2),
            "cats": bd.get("cats", []),
            "escal": bd.get("escal_flag", False),
            "arc": bd.get("arc_flag", False),
        }


def flatten(messages: list[dict]) -> str:
    cat_counts: dict[str, int] = {}
    pred_msgs: list[dict] = []
    parts: list[str] = []
    total_msgs = 0

    for conv_idx, msg in enumerate(messages):
        norm = normalise(msg.get("text_raw", ""))
        if not norm:
            continue
        total_msgs += 1
        role = "[PRED]" if msg.get("is_pred") else "[USER]"

        if msg.get("is_pred"):
            matches = _RULE_ENGINE.match(norm, already_normalised=True)
            max_sev = max((m.severity for m in matches), default=0)
            cats_hit = {m.category for m in matches}
            for cat in cats_hit:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            pred_msgs.append({
                "conv_idx": conv_idx,
                "max_sev": max_sev,
                "cats": cats_hit,
                "matches": matches,
            })
            seen = set()
            tags = []
            for m in matches:
                key = (m.category, m.severity, m.method)
                if key not in seen:
                    seen.add(key)
                    tags.append(f"[RULE:{m.category}:{m.severity}:{m.method}]")
            rule_prefix = (" ".join(tags) + " ") if tags else ""
            parts.append(f"{role} {rule_prefix}{norm}")
        else:
            parts.append(f"{role} {norm}")

    _, level, bd = risk_score(cat_counts, pred_msgs, total_msgs)
    esc_tok = "[ESC:1]" if bd["escal"] > 0 else "[ESC:0]"
    arc_tok = "[ARC:1]" if bd["arc"] > 0 else "[ARC:0]"
    dom_tok = "[DOM:1]" if bd["dominance"] > 0 else "[DOM:0]"
    cats_str = ",".join(sorted(cat_counts.keys())) if cat_counts else "none"
    prefix = f"[RISK:{level}] [CATS:{cats_str}] {esc_tok} {arc_tok} {dom_tok}"
    body = " [SEP] ".join(parts)
    return f"{prefix} {body}" if body else prefix


def load_predator_ids_txt(path: str) -> set[str]:
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def load_problem1_gt(path: str) -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            conv_id, author_id = parts[0].strip(), parts[1].strip()
            gt.setdefault(conv_id, set()).add(author_id)
    return gt


XML_PROGRESS_EVERY = 500


def _tag_name(elem) -> str:
    tag = elem.tag
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _child_text(elem, name: str) -> str:
    for child in elem:
        if _tag_name(child) == name:
            return (child.text or "").strip()
    return ""


def parse_xml(xml_path: str, predator_ids: Optional[set[str]] = None, test_gt1: Optional[dict[str, set[str]]] = None) -> list[dict]:
    if (predator_ids is None) == (test_gt1 is None):
        raise ValueError("Provide exactly one of predator_ids or test_gt1")

    xml_path = str(xml_path)
    records = []
    n_conv = 0
    n_pos = 0

    print(f"  streaming: {Path(xml_path).name}", flush=True)

    for _event, conv in ET.iterparse(xml_path, events=("end",)):
        if _tag_name(conv) != "conversation":
            continue

        conv_id = conv.get("id", "").strip()
        local_pred = predator_ids if predator_ids is not None else test_gt1.get(conv_id, set())
        messages = []
        is_grooming = False

        for msg in conv:
            if _tag_name(msg) != "message":
                continue
            author = _child_text(msg, "author")
            text_raw = _child_text(msg, "text")
            is_pred = author in local_pred
            if is_pred:
                is_grooming = True
            messages.append({"author": author, "text_raw": text_raw, "is_pred": is_pred})

        label = int(is_grooming)
        n_conv += 1
        n_pos += label
        records.append({
            "conversation_id": conv_id,
            "text": flatten(messages),
            "label": label,
        })

        if n_conv % XML_PROGRESS_EVERY == 0:
            print(f"    parsed {n_conv:,} conversations | pos={n_pos:,}", flush=True)

        conv.clear()

    print(f"  parsed {n_conv:,} conversations | pos={n_pos:,}", flush=True)
    return records

def to_records(records: list[dict]) -> list[dict]:
    return [{"conversation_id": r["conversation_id"], "text": r["text"], "label": r["label"]} for r in records]


def print_stats(ds: dict[str, list[dict]]) -> None:
    for split, data in ds.items():
        n = len(data)
        pos = sum(r["label"] for r in data)
        pct = 100 * pos / n if n else 0.0
        print(f"  {split:6s}: {n:6d} total | {pos:5d} pos ({pct:.1f}%) | {n - pos:5d} neg")


# ---------------------------------------------------------------------------
# Default preprocessing config
# ---------------------------------------------------------------------------

TRAIN_XML = "data/train/pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
TRAIN_PRED = "data/train/pan12-sexual-predator-identification-training-corpus-predators-2012-05-01.txt"
TEST_XML = "data/test/pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"
TEST_GT1 = "data/test/pan12-sexual-predator-identification-groundtruth-problem1.txt"
OUTPUT = "pan12_dataset"
VAL_RATIO = 0.10
SEED = 42


def _require_file(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing required file: {path}")



def _split_train_val(records: list[dict], labels: list[int]) -> tuple[list[dict], list[dict]]:
    class_counts = {0: labels.count(0), 1: labels.count(1)}
    can_stratify = min(class_counts.values()) >= 2
    stratify = labels if can_stratify else None
    if not can_stratify:
        print("  warning: stratified split disabled; too few samples in at least one class", flush=True)
    return train_test_split(
        records,
        test_size=VAL_RATIO,
        stratify=stratify,
        random_state=SEED,
    )

def preprocess_pan12() -> None:
    if train_test_split is None:
        raise RuntimeError("scikit-learn is required for train/val split")

    _require_file(TRAIN_XML)
    _require_file(TRAIN_PRED)
    _require_file(TEST_XML)
    _require_file(TEST_GT1)

    print("Loading training predator IDs...", flush=True)
    pred_ids = load_predator_ids_txt(TRAIN_PRED)

    print("Parsing training XML...", flush=True)
    train_all = parse_xml(TRAIN_XML, predator_ids=pred_ids)
    labels = [r["label"] for r in train_all]

    train_rec, val_rec = _split_train_val(train_all, labels)

    print("Loading test ground truth...", flush=True)
    gt1 = load_problem1_gt(TEST_GT1)

    print("Parsing test XML...", flush=True)
    test_rec = parse_xml(TEST_XML, test_gt1=gt1)

    ds = {
        "train": to_records(train_rec),
        "val": to_records(val_rec),
        "test": to_records(test_rec),
    }

    print("\nDataset statistics:")
    print_stats(ds)

    out = Path(OUTPUT)
    out.mkdir(parents=True, exist_ok=True)

    for split, records in ds.items():
        path = out / f"{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  wrote {path} ({path.stat().st_size // 1024} KB)")


def main() -> None:
    preprocess_pan12()


if __name__ == "__main__":
    main()
