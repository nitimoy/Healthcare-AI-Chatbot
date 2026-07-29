"""
app/chatbot/guardrails.py
──────────────────────────
Medical safety guardrails — detect and intercept dangerous requests.

Responsibility
──────────────
Before any LLM call, scan the user query for patterns that indicate
the user may be in a medical emergency or seeking advice that could
cause harm (self-medication, dosage queries, emergency symptoms).

If a dangerous pattern is detected, the chatbot bypasses the normal
RAG pipeline and returns a pre-written safety response that encourages
the user to seek immediate professional care.

Design decisions
────────────────
1.  Keyword-first (zero LLM cost):
    Pattern matching is O(n) string operations — effectively free.
    No embedding, no API call, no hallucination risk.

2.  Two categories of danger:
    a. EMERGENCY — chest pain, stroke, heart attack, suicide, overdose.
       → Immediate 999/911 response.
    b. UNSAFE REQUEST — diagnose me, what medicine, dosage, prescription.
       → Redirect to professional without providing advice.

3.  Whole-word matching via regex:
    "stroke" as a substring would match "keystroke". We use word boundaries
    (\\b) to prevent false positives on clinical educational text.
    e.g. "stroke prevention" is NOT flagged — it is a health topic.
    "I think I'm having a stroke" IS flagged.

4.  The guardrail intercepts at the generator level:
    `is_emergency(query)` and `is_unsafe(query)` are checked before
    retrieval so no vector search or API call is made for dangerous inputs.
    This keeps latency near zero for safety interceptions.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Pattern lists
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that suggest a life-threatening emergency requiring immediate care.
_EMERGENCY_PATTERNS: list[str] = [
    r"\bchest\s+pain\b",
    r"\bchest\s+tightness\b",
    r"\bhaving\s+a\s+stroke\b",
    r"\bstroke\s+symptoms\b",
    r"\bheart\s+attack\b",
    r"\bcardiac\s+arrest\b",
    r"\bcan'?t\s+breathe\b",
    r"\bcannot\s+breathe\b",
    r"\bstopped\s+breathing\b",
    r"\b(struggling|difficulty|hard|trouble)\s+to\s+breathe\b",
    r"\b(baby|child|infant|toddler)\b.*\b(struggling|difficulty|hard|trouble|gasping)\s+to\s+breathe\b",
    r"\b(struggling|difficulty|hard|trouble|gasping)\s+to\s+breathe.*\b(baby|child|infant|toddler)\b",
    r"\bunresponsive\b",
    r"\bunconscious\b",
    r"\b(not|won't)\s+wake\s+up\b",
    r"\bcollapsed\s+and\s+not\s+responding\b",
    r"\banaphylax\w*\b",
    r"\bsevere\s+allergic\s+reaction\b",
    r"\b(throat|tongue|face|lips)\s+(closing|swelling|swollen)\b",
    r"\b(can't|cannot|unable\s+to)\s+stop\s+bleeding\b",
    r"\b(blood\s+is\s+spurting|heavy\s+bleeding)\b",
    r"\bpoison\w*\b",
    r"\bpoisoning\b",
    r"\bingested\s+(poison|toxic|bleach|chemical|cleaner)s?\b",
    r"\bpregnancy\s+emergency\b",
    r"\bsuicid\w+\b",
    r"\bkill\s+(my)?self\b",
    r"\bself[- ]?harm\b",
    r"\boverdos\w*\b",
    r"\bwant\s+to\s+die\b",
    r"\bend(ing)?\s+\w+\s+life\b",
]

# Patterns that request medication/treatment modifications (stop, increase, decrease, double, skip, substitute).
_MEDICATION_MOD_PATTERNS: list[str] = [
    r"\b(can|should)\s+i\s+(stop|discontinue|pause|quit|increase|decrease|double|lower|change|reduce|skip)\s+(taking|my|the)?\s*(medicine|medication|drug|pill|dose|dosage|chemotherapy|insulin|blood\s+pressure|antibiotics?|steroids?|inhaler)\b",
    r"\b(stop\s+taking|reduce\s+dose|increase\s+dose|skip\s+today'?s\s+dose|double\s+my\s+dose|change\s+my\s+medication)\b",
    r"\b(stop|discontinue|increase|decrease|double|lower|reduce|skip)\s+(my\s+)?(chemotherapy|insulin|blood\s+pressure\s+med\w*|steroids?|inhaler|dose|dosage)\b",
    r"\b(my\s+doctor|doctor\s+prescribed)\b.*\b(should|can)\s+i\s+(stop|discontinue|pause|change|reduce|skip)\b",
    r"\bshould\s+i\s+increase\b",
    r"\bshould\s+i\s+double\b",
    r"\bshould\s+i\s+lower\b",
    r"\bcan\s+i\s+stop\b",
    r"\bshould\s+i\s+stop\b",
]

# Patterns that request personalized disease risk predictions.
_PREDICTION_PATTERNS: list[str] = [
    r"\b(will|am\s+i\s+going\s+to|what\s+are\s+my\s+chances\s+of)\s+(get|develop|have)\s+(cancer|diabetes|stroke|heart\s+disease|dementia)\b",
    r"\bwill\s+i\s+develop\b",
    r"\bwill\s+i\s+get\b",
]

# Patterns that request clinical decisions or personalized medication advice.
_UNSAFE_REQUEST_PATTERNS: list[str] = [
    r"\bdiagnose\s+me\b",
    r"\bwhat\s+(medicine|medication|drug|pill|tablet)s?\s+(should|do)\s+i\b",
    r"\bwhich\s+(medicine|medication|drug|pill|tablet)s?\s+(should|can)\s+i\s+take\b",
    r"\bwhich\s+medicine\s*\?\b",
    r"\bwhat\s+dose\b",
    r"\bdosage\s+of\b",
    r"\bhow\s+much\s+(of\s+)?(medicine|medication|drug)\b",
    r"\bprescri(be|ption)\b",
    r"\bam\s+i\s+(sick|ill|dying|infected)\b",
    r"\bdo\s+i\s+have\s+(a\s+)?(disease|cancer|diabetes|stroke|infection)\b",
    r"\btreat\s+my\s+(disease|cancer|infection|condition)\b",
    r"\bshould\s+i\s+take\b",
    r"\bwhich\s+antibiotic\b",
    r"\b(replace|substitute|swap)\s+.*(inhaler|medication|medicine|prescription|drug|steroids?)\b",
    r"\b(can|should)\s+i\s+(give|use|take)\s+(herbal|herbs?|natural|supplement|homeopathy|alternative)s?\s+(instead\s+of|rather\s+than|to\s+replace)\b",
    r"\bherbal\s+medicine\s+instead\s+of\b",
    r"\bcan\s+i\s+take\s+(ibuprofen|aspirin|paracetamol|tylenol|advil|naproxen|steroids?)\s+(with|and|while\s+on)\s+(blood\s+pressure|bp|hypertension|diabetes|anticoagulant|blood\s+thinner|medication|medicine)\b",
]

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compiled regex (compile once at module load for speed)
# ─────────────────────────────────────────────────────────────────────────────

_EMERGENCY_RE = re.compile("|".join(_EMERGENCY_PATTERNS), flags=re.IGNORECASE)
_MEDICATION_MOD_RE = re.compile("|".join(_MEDICATION_MOD_PATTERNS), flags=re.IGNORECASE)
_PREDICTION_RE = re.compile("|".join(_PREDICTION_PATTERNS), flags=re.IGNORECASE)
_UNSAFE_RE = re.compile("|".join(_UNSAFE_REQUEST_PATTERNS), flags=re.IGNORECASE)

_DIET_RE = re.compile(
    r"\b(diet|keto|ketogenic|fasting|intermittent\s+fasting|meal\s+plan|weight\s+loss|dash|mediterranean|vegan|low[- ]carb)\b",
    flags=re.IGNORECASE,
)

_CONDITION_RE = re.compile(
    r"\b(diabetes|diabetic|kidney|renal|hypertension|high\s+blood\s+pressure|heart\s+disease|cholesterol|cancer|pregnancy|pregnant|breastfeeding|child|children|infant|infants)\b",
    flags=re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Pre-written safety responses (no LLM call — deterministic)
# ─────────────────────────────────────────────────────────────────────────────

_EMERGENCY_RESPONSE = """\
🚨 **MEDICAL EMERGENCY DETECTED**

**Please call emergency services immediately:**
- 🇺🇸 **USA / Canada:** 911
- 🇬🇧 **UK:** 999
- 🇪🇺 **Europe:** 112
- 🇮🇳 **India:** 112

If you or someone else is in immediate danger, **do not wait — call now.**

This chatbot cannot provide emergency medical guidance. Please seek help \
from trained emergency medical personnel right away.

---
⚠️ *If you are experiencing thoughts of self-harm or suicide, please contact \
a crisis helpline:*
- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/
- **Crisis Text Line (US):** Text HOME to 741741
"""

_MEDICATION_MOD_RESPONSE = """\
⚠️ **I cannot advise whether you should stop, increase, decrease, or modify a prescribed medication or treatment.**

Changing your treatment plan (such as stopping blood pressure medication, altering insulin doses, or discontinuing chemotherapy) can have severe health risks. 

**What you should do:**
- 👨‍⚕️ Contact your prescribing doctor or oncologist immediately before altering any treatment.
- 💊 Speak with a licensed pharmacist if you have questions about side effects or dosing.

---
⚠️ *This chatbot provides educational health information only and cannot provide treatment advice.*
"""

_PREDICTION_RESPONSE = """\
⚠️ **I cannot predict whether an individual will develop a specific medical condition.**

Disease risk depends on a complex combination of genetics, personal medical history, lifestyle factors, and clinical lab results that require evaluation by a physician.

**What I recommend instead:**
- 👨‍⚕️ Discuss your personal health history and family risk factors with your primary care provider.
- 🩺 Schedule routine preventive screenings appropriate for your age and background.

I can, however, provide **general educational information** on the known risk factors and prevention strategies for specific conditions. How can I help you with a general educational question?

---
⚠️ *This chatbot provides educational health information only and cannot assess personal medical risk.*
"""

_UNSAFE_REQUEST_RESPONSE = """\
⚠️ **I'm not able to answer that question.**

This chatbot is designed to provide **general health education only**. \
It cannot:
- Diagnose diseases or medical conditions
- Recommend or prescribe medications or dosages
- Provide personalised treatment plans
- Replace the advice of a licensed healthcare professional

**What I recommend instead:**
- 👨‍⚕️ Consult your primary care physician or a specialist
- 🏥 Visit an urgent care centre if needed soon
- 📞 Call your local health advice line

I'm happy to answer **general questions** about symptoms, healthy lifestyles, \
nutrition, preventive care, or how conditions are typically managed. \
How can I help you with a general health question?

---
⚠️ *This chatbot provides educational healthcare information only and should \
not replace professional medical advice.*
"""

_MULTI_CONDITION_DIET_RESPONSE = """\
⚠️ **I cannot recommend specific diets or meal plans for individuals with medical conditions or special health needs.**

Managing nutrition during pregnancy, breastfeeding, childhood, or when dealing with chronic conditions (such as diabetes, kidney disease, or hypertension) requires individualised clinical assessment. Standard diets (such as keto or fasting) can affect blood sugar, kidney filtration, and fetal/child development.

**What I recommend instead:**
- 👨‍⚕️ Consult your primary physician or a Registered Dietitian (RD)
- 🩺 Work with a certified specialist for a personalized nutrition plan

I can, however, provide **general educational information** on the principles of these diets or individual health conditions. How can I help you with a general educational topic?

---
⚠️ *This chatbot provides educational healthcare information only and should not replace professional medical advice.*
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def is_emergency(query: str) -> bool:
    """Return True if the query contains emergency medical indicators."""
    matched = bool(_EMERGENCY_RE.search(query))
    if matched:
        logger.warning("GUARDRAIL [EMERGENCY] triggered for query: %r", query[:100])
    return matched


def is_medication_mod_request(query: str) -> bool:
    """Return True if query asks to stop, increase, decrease, or modify treatment."""
    matched = bool(_MEDICATION_MOD_RE.search(query))
    if matched:
        logger.warning("GUARDRAIL [MEDICATION_MOD] triggered for query: %r", query[:100])
    return matched


def is_prediction_request(query: str) -> bool:
    """Return True if query asks for personalized disease risk prediction."""
    matched = bool(_PREDICTION_RE.search(query))
    if matched:
        logger.warning("GUARDRAIL [PREDICTION] triggered for query: %r", query[:100])
    return matched


def is_unsafe_request(query: str) -> bool:
    """Return True if the query requests clinical decisions (diagnose, prescribe, etc.)."""
    matched = bool(_UNSAFE_RE.search(query))
    if matched:
        logger.warning("GUARDRAIL [UNSAFE_REQUEST] triggered for query: %r", query[:100])
    return matched


def is_multi_condition_diet_request(query: str) -> bool:
    """Return True if query seeks personalized diet recommendations for high-risk groups or multiple conditions."""
    has_diet = bool(_DIET_RE.search(query))
    conditions = set(c.lower() for c in _CONDITION_RE.findall(query))
    matched = has_diet and len(conditions) >= 1
    if matched:
        logger.warning("GUARDRAIL [MULTI_CONDITION_DIET] triggered for query: %r", query[:100])
    return matched


def check_and_respond(query: str) -> str | None:
    """Run all guardrail checks and return a safety response if triggered."""
    if is_emergency(query):
        return _EMERGENCY_RESPONSE
    if is_medication_mod_request(query):
        return _MEDICATION_MOD_RESPONSE
    if is_prediction_request(query):
        return _PREDICTION_RESPONSE
    if is_multi_condition_diet_request(query):
        return _MULTI_CONDITION_DIET_RESPONSE
    if is_unsafe_request(query):
        return _UNSAFE_REQUEST_RESPONSE
    return None
