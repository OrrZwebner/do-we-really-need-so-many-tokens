"""Post-hoc grounding checks on the model's answer.

The system prompt tells the model to quote numbers verbatim and never invent a
citation. This module checks whether it actually did, by comparing the answer
against the passages that were really retrieved.

It is a safety net, not a gate. Two honest reasons a number can be absent from
the source text:

* the model computed a patient-specific dose from a per-kg figure and a weight
  (15 mg/kg x 12 kg = 180 mg) — the product is derived, not quoted;
* the answer restates a number in different units.

So findings are reported as "verify this against the source", never as
"the model lied". What they reliably catch is the dangerous case: a dose or a
page number that appears in the answer and nowhere in the retrieved text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..indexing.store import Hit

# --------------------------------------------------------------------------
# Dose extraction
# --------------------------------------------------------------------------

_UNIT_ALIASES = {
    "µg": "mcg", "μg": "mcg", "ug": "mcg",
    "ml": "ml", "milliliter": "ml", "millilitre": "ml",
    "unit": "unit", "units": "unit", "iu": "unit",
    "gram": "g", "grams": "g",
    "milligram": "mg", "milligrams": "mg",
    "microgram": "mcg", "micrograms": "mcg",
}

_UNIT = r"(?:mg|mcg|µg|μg|ug|g|kg|mL|ml|L|units?|IU|mEq|mmol|mOsm)"

# "15 mg/kg", "0.1 mg/kg/dose", "6-8 mg/kg/min"
_PER_KG = re.compile(
    rf"(\d[\d.,]*(?:\s*[-–]\s*\d[\d.,]*)?)\s*({_UNIT})\s*/\s*kg"
    rf"(?:\s*/\s*(?:dose|day|d|hr|h|min))?",
    re.I,
)
# A bare dose amount: "500 mg", "2 g", "10 mL"
_ABSOLUTE = re.compile(rf"(\d[\d.,]*(?:\s*[-–]\s*\d[\d.,]*)?)\s*({_UNIT})\b", re.I)


@dataclass(frozen=True)
class DoseClaim:
    raw: str
    normalised: str
    per_kg: bool


def _normalise_number(value: str) -> str:
    value = value.replace(",", "").replace("–", "-")
    value = re.sub(r"\s*-\s*", "-", value).strip()
    # 0.50 and .5 and 0.5 should all compare equal.
    parts = []
    for token in value.split("-"):
        try:
            number = float(token)
            parts.append(f"{number:g}")
        except ValueError:
            parts.append(token)
    return "-".join(parts)


def _normalise_unit(unit: str) -> str:
    lowered = unit.lower()
    return _UNIT_ALIASES.get(lowered, lowered)


def extract_dose_claims(text: str) -> list[DoseClaim]:
    """Every dose-shaped number in ``text``, normalised for comparison."""
    claims: dict[str, DoseClaim] = {}
    spans: list[tuple[int, int]] = []

    for match in _PER_KG.finditer(text):
        number, unit = _normalise_number(match.group(1)), _normalise_unit(match.group(2))
        key = f"{number}{unit}/kg"
        claims.setdefault(key, DoseClaim(match.group(0).strip(), key, True))
        spans.append(match.span())

    for match in _ABSOLUTE.finditer(text):
        # Skip anything already consumed by a per-kg match.
        if any(lo <= match.start() < hi for lo, hi in spans):
            continue
        unit = _normalise_unit(match.group(2))
        if unit == "kg":  # a patient weight, not a dose
            continue
        number = _normalise_number(match.group(1))
        key = f"{number}{unit}"
        claims.setdefault(key, DoseClaim(match.group(0).strip(), key, False))

    return list(claims.values())


# --------------------------------------------------------------------------
# Citation extraction
# --------------------------------------------------------------------------

_CHAPTER_REF = re.compile(r"\b(?:Ch\.?|Chapter|פרק)\s*(\d+(?:\.\d+)?)", re.I)
_PAGE_REF = re.compile(r"\bpp?\.\s*(\d+)(?:\s*[-–]\s*(\d+))?", re.I)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class GroundingReport:
    dose_claims: int = 0
    ungrounded_doses: list[str] = field(default_factory=list)
    ungrounded_chapters: list[str] = field(default_factory=list)
    ungrounded_pages: list[str] = field(default_factory=list)
    passages_used: int = 0
    searched: bool = True

    @property
    def clean(self) -> bool:
        return not (
            self.ungrounded_doses or self.ungrounded_chapters or self.ungrounded_pages
        ) and self.searched

    def warning_text(self) -> str | None:
        """A short note to show the clinician, or None when nothing is off."""
        lines: list[str] = []
        if not self.searched:
            lines.append(
                "This answer was produced WITHOUT searching the textbook. "
                "Treat it as unsourced."
            )
        if self.ungrounded_doses:
            shown = ", ".join(self.ungrounded_doses[:6])
            lines.append(
                f"These figures are not verbatim in the retrieved passages: {shown}. "
                "Expected for a dose computed from a weight — otherwise check the "
                "source before acting on them."
            )
        if self.ungrounded_chapters:
            lines.append(
                "Chapter(s) cited but not among the retrieved passages: "
                + ", ".join(self.ungrounded_chapters[:6])
            )
        if self.ungrounded_pages:
            lines.append(
                "Page(s) cited but not among the retrieved passages: "
                + ", ".join(self.ungrounded_pages[:6])
            )
        return "\n".join(f"- {line}" for line in lines) if lines else None


def check_answer(answer: str, retrieved: list[Hit], searched: bool = True) -> GroundingReport:
    """Compare an answer against the passages actually retrieved for it."""
    report = GroundingReport(passages_used=len(retrieved), searched=searched)
    if not answer.strip():
        return report

    corpus = "\n".join(hit.text for hit in retrieved)
    corpus_claims = {c.normalised for c in extract_dose_claims(corpus)}

    for claim in extract_dose_claims(answer):
        report.dose_claims += 1
        if claim.normalised not in corpus_claims:
            report.ungrounded_doses.append(claim.raw)

    known_chapters = {str(h.chapter_number) for h in retrieved if h.chapter_number}
    for match in _CHAPTER_REF.finditer(answer):
        if known_chapters and match.group(1) not in known_chapters:
            ref = f"Ch. {match.group(1)}"
            if ref not in report.ungrounded_chapters:
                report.ungrounded_chapters.append(ref)

    known_pages: set[int] = set()
    for hit in retrieved:
        if hit.page_start and hit.page_end:
            known_pages.update(range(hit.page_start, hit.page_end + 1))
    for match in _PAGE_REF.finditer(answer):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        if known_pages and not (set(range(start, end + 1)) & known_pages):
            ref = f"p. {start}" if start == end else f"pp. {start}-{end}"
            if ref not in report.ungrounded_pages:
                report.ungrounded_pages.append(ref)

    return report
