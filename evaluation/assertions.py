import re
from typing import Iterable


CONCEPT_PATTERNS = {
    "final sale does not block damaged-item review": [
        r"final[- ]sale.*(?:still|does not|doesn't).*?(?:damag|defect|wrong|incorrect)",
        r"(?:damag|defect|wrong|incorrect).*final[- ]sale.*(?:review|eligible)",
    ],
    "report within 7 days": [r"7 (?:calendar )?days"],
    "human review before approval": [
        r"(?:human|support specialist).*review",
        r"review.*before.*approval|(?:cannot|can't|must not|not).*approve|approval.*(?:human|review)",
    ],
    "Canada is supported": [r"ship.*canada", r"canada.*supported", r"only.*canada"],
    "5–9 business days after dispatch": [r"5\s*[–-]\s*9 business days.*after dispatch"],
    "duties or taxes are not prepaid": [r"dut(?:y|ies)|tax", r"not prepaid|recipient.*responsible"],
    "shipping to Germany is not currently available": [r"germany.*(?:not|isn't).*available", r"only.*canada"],
    "the order is cancelled": [r"cancelled|canceled"],
    "it will not be shipped": [r"will not be shipped|won't be shipped|not be shipped"],
    "shipped with Canada Post": [r"shipped.*canada post|canada post"],
    "delivery estimate is unavailable": [r"estimate.*(?:unavailable|not currently available)|not.*estimate"],
    "order was not found": [r"couldn't find|could not find|not found"],
    "check the order ID or contact support": [r"check.*order id|contact support"],
    "no lifetime warranty": [r"(?:does not|doesn't|no).*lifetime warranty"],
    "bags have 2 years": [r"bags?.*2 years|backpacks?.*2 years"],
    "drinkware and travel accessories have 1 year": [r"drinkware.*1 year", r"travel accessories.*1 year"],
    "migration note is not authoritative": [r"migration.*not authoritative|draft.*not authoritative|not authoritative.*policy"],
    "standard policy is 30 days unless a valid exception applies": [r"30 calendar days|30 days"],
    "the agent cannot approve a return": [r"cannot approve|can't approve|no approval action"],
    "the supplied information is insufficient": [r"insufficient|not enough information|don't have enough information"],
    "human confirmation": [r"human confirmation|human support|support specialist"],
    "current official sources conflict": [r"conflict|inconsistent"],
    "one says hand-wash the body": [r"hand[- ]wash.*body|body.*hand[- ]wash"],
    "one says all components are dishwasher safe": [r"all components.*dishwasher safe|dishwasher safe.*all components"],
    "human confirmation or safest interim guidance": [r"human confirmation|support specialist|safest interim|hand[- ]wash"],
    "weather delay": [r"weather delay"],
    "August 20, 2026": [r"august 20,? 2026"],
    "pending": [r"pending"],
    "within 30 minutes": [r"30[- ]minute|within.*30 minutes"],
    "cannot complete cancellation": [r"cannot complete|can't complete|have not cancelled|not cancelled"],
    "human review": [r"human|support specialist"],
    "processing": [r"processing"],
    "cannot guarantee address change": [r"cannot guarantee.*address|can't guarantee.*address"],
    "not completed": [r"have not|has not|not.*completed|cannot complete|can't complete|not changed"],
    "items and quantities cannot be edited after checkout": [r"items.*quantit.*cannot be edited|cannot.*edit.*after checkout"],
    "joining after purchase does not extend return window": [r"joining.*after.*does not extend|after.*placing.*does not extend|must.*active.*when.*order"],
    "do not share complete gift-card code": [r"do not.*(?:complete|full).*gift[- ]?card code|must not.*share.*gift[- ]?card code|don't.*share.*gift[- ]?card code"],
}


def contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


REQUIRE_ALL_CONCEPTS = {
    "duties or taxes are not prepaid",
    "drinkware and travel accessories have 1 year",
    "human review before approval",
}


def concept_matches(text: str, concept: str) -> bool:
    patterns = CONCEPT_PATTERNS.get(concept)
    if not patterns:
        return contains(text, concept)

    matches = [
        bool(re.search(pattern, text, flags=re.I | re.S))
        for pattern in patterns
    ]

    if concept in REQUIRE_ALL_CONCEPTS:
        return all(matches)
    return any(matches)


def any_refusal(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:can't|cannot|won't|will not|not able to|refuse|do not provide)\b",
            text,
            flags=re.I,
        )
    )


def sources_set(responses: Iterable) -> set[str]:
    return {
        source.get("filename", "")
        for response in responses
        for source in response.sources
    }
