"""Stage 4: the planner. Edited TEXT in, cuts derived elsewhere.

The model is given the packed transcript, the learned style profile and worked
examples, and returns three things: the transcript with material deleted, insert
markers anchored to exact word strings, and a plain-English strategy. It never
returns a number in seconds -- there is nowhere in this module to put one (hard
rule 12). An insert marker is resolved here to a WORD INDEX; the second it
becomes a time is inside `derive_cuts`, from the aligner's own numbers.

Everything the model returns is checked before it is allowed to exist as a plan:

  * paraphrase  - `diff_align` must find every edited word in the source
  * cut ratio   - inside the profile band, through the real cut engine
  * inserts     - every `after_text` must be locatable in the words that SURVIVE

A failure is fed back to the model once, in its own words, then the run stops.
Silently repairing a broken plan would hide exactly the failure mode this
architecture exists to prevent.
"""

from __future__ import annotations

# See derive_cuts: makes `python helpers/plan_edit.py` work as well as imports.
if __package__ in (None, ""):  # pragma: no cover - script entry only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    __package__ = "helpers"

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import config, derive_cuts, diff_align, gemini_client, paths, textnorm
from .words import WordsDoc

SCHEMA = "zeta.edit_plan.v1"

# The spec's prompt, kept close to the wording that was tested. Changes here
# change the edit, so treat it as an interface, not as copy.
SYSTEM_PROMPT = """\
You are the editor of Zeta, a Darija news channel (Moroccan Arabic, code-switching
with French and English). You receive the verbatim phrase transcript of a raw take
with gap durations, the style profile, and worked examples from past episodes.

Return JSON:
  {"kept_text": "<full transcript with removed material deleted, nothing paraphrased>",
   "inserts": [{"after_text": "<exact 3 to 6 words before which the visual should appear>",
                "claim": "<what must be visible>",
                "entity": "<company/person/source>",
                "prefer_source": "<owner's own page or post if known>"}],
   "strategy": "<4 to 8 sentences in plain English>"}

Rules:
* Never change, translate, re-spell or reorder a word you keep. Copy kept words
  exactly as they appear in the transcript, including French and English words.
* Delete whole phrases where possible.
* Remove the fillers listed in filler_policy.remove, false starts, and earlier
  retakes (keep the last complete take).
* Keep the hook and the sentence that names the source.
* Respect cut_ratio within the stated tolerance.
* Place insert markers only where the profile's triggers apply. `after_text` must
  be copied verbatim from the transcript and must be text you KEPT.
* Never output timestamps. Timing is not your job and any number in seconds you
  produce will be discarded.
"""

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "kept_text": {"type": "string"},
        "inserts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "after_text": {"type": "string"},
                    "claim": {"type": "string"},
                    "entity": {"type": "string"},
                    "prefer_source": {"type": "string"},
                },
                "required": ["after_text", "claim", "entity"],
                "propertyOrdering": ["after_text", "claim", "entity", "prefer_source"],
            },
        },
        "strategy": {"type": "string"},
    },
    "required": ["kept_text", "inserts", "strategy"],
    "propertyOrdering": ["kept_text", "inserts", "strategy"],
}

# An anchor shorter than this resolves onto whatever common word it happens to
# contain, so the visual lands in the wrong sentence.
MIN_ANCHOR_TOKENS = 2


class PlanError(RuntimeError):
    """The planner returned something that cannot become an edit."""

    def __init__(self, message: str, failures: Sequence[str] = ()):
        self.failures = list(failures)
        super().__init__(message)


@dataclass
class Insert:
    """One visual, anchored to a word index -- never to a timestamp."""

    after_text: str
    claim: str
    entity: str
    prefer_source: str | None = None
    trigger_word_index: int = -1
    trigger_word: str = ""
    anchor_span: tuple[int, int] = (-1, -1)
    ambiguous: bool = False

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "after_text": self.after_text,
            "claim": self.claim,
            "entity": self.entity,
            "trigger_word_index": self.trigger_word_index,
            "trigger_word": self.trigger_word,
            "anchor_span": list(self.anchor_span),
        }
        if self.prefer_source:
            d["prefer_source"] = self.prefer_source
        if self.ambiguous:
            d["ambiguous"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Insert":
        return cls(after_text=d.get("after_text", ""), claim=d.get("claim", ""),
                   entity=d.get("entity", ""), prefer_source=d.get("prefer_source"),
                   trigger_word_index=int(d.get("trigger_word_index", -1)),
                   trigger_word=d.get("trigger_word", ""),
                   anchor_span=tuple(d.get("anchor_span", (-1, -1))),  # type: ignore[arg-type]
                   ambiguous=bool(d.get("ambiguous", False)))


@dataclass
class EditPlan:
    source_name: str
    kept_text: str
    strategy: str
    inserts: list[Insert] = field(default_factory=list)
    cut_ratio: float = 0.0
    target_cut_ratio: float | None = None
    within_band: bool = True
    attempts: int = 1
    warnings: list[str] = field(default_factory=list)
    needs_visual_check: list[dict] = field(default_factory=list)
    words_path: str | None = None
    profile_path: str | None = None
    confirmed: bool = False

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "source": self.source_name,
            "kept_text": self.kept_text,
            "strategy": self.strategy,
            "inserts": [i.to_dict() for i in self.inserts],
            "derived": {
                "cut_ratio": round(self.cut_ratio, 4),
                "target_cut_ratio": self.target_cut_ratio,
                "within_band": self.within_band,
                "needs_visual_check": self.needs_visual_check,
            },
            "meta": {
                "attempts": self.attempts,
                "warnings": self.warnings,
                "words": self.words_path,
                "style_profile": self.profile_path,
                "confirmed": self.confirmed,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EditPlan":
        derived = d.get("derived", {})
        meta = d.get("meta", {})
        return cls(
            source_name=d.get("source", ""), kept_text=d.get("kept_text", ""),
            strategy=d.get("strategy", ""),
            inserts=[Insert.from_dict(i) for i in d.get("inserts", [])],
            cut_ratio=float(derived.get("cut_ratio", 0.0)),
            target_cut_ratio=derived.get("target_cut_ratio"),
            within_band=bool(derived.get("within_band", True)),
            needs_visual_check=list(derived.get("needs_visual_check", [])),
            attempts=int(meta.get("attempts", 1)), warnings=list(meta.get("warnings", [])),
            words_path=meta.get("words"), profile_path=meta.get("style_profile"),
            confirmed=bool(meta.get("confirmed", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EditPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                     encoding="utf-8")
        return p


# -------- prompt --------------------------------------------------------------


def packed_view(doc: WordsDoc, *, break_gap_s: float = 0.5) -> str:
    """The planner's reading view, from the one packer.

    The prompt tells the model it is getting gap durations and language tags, so
    it has to be the real packed format from `pack_transcripts`, not a lookalike.
    A second format here would drift from the one `zeta transcribe` writes and
    the model would be reading a view nobody else has seen.
    """
    from . import pack_transcripts

    return pack_transcripts.pack_doc(doc, silence_s=break_gap_s)

def build_prompt(packed_text: str, profile: dict, few_shot: str = "",
                 feedback: str | None = None) -> str:
    """Assemble the user half of the planner prompt."""
    tolerance = profile.get("cut_ratio_tolerance", derive_cuts.CUT_RATIO_TOLERANCE)
    parts = [
        "STYLE PROFILE (learned from past Zeta episodes):",
        # `examples` is a filename and `meta` is provenance: neither is an
        # instruction to the editor, and both cost prompt budget.
        json.dumps({k: v for k, v in profile.items() if k not in ("examples", "meta")},
                   ensure_ascii=False, indent=1),
        f"\nTarget cut_ratio {profile.get('cut_ratio', 'unknown')} "
        f"within +/-{tolerance}.",
    ]
    if few_shot.strip():
        parts += ["\nWORKED EXAMPLES FROM PAST EPISODES:", few_shot.strip()]
    parts += ["\nRAW TAKE (verbatim, phrase per line, gap is the silence before the phrase):",
              packed_text.strip()]
    if feedback:
        # The retry carries the exact failure. A vague "try again" gets a
        # rephrased version of the same broken answer.
        parts += ["\nYOUR PREVIOUS ANSWER WAS REJECTED:", feedback,
                  "Fix exactly that and return the same JSON shape."]
    return "\n".join(parts)


# -------- validation ----------------------------------------------------------


def _kept_indices(kept_spans: Sequence) -> list[int]:
    return [i for s in kept_spans for i in range(s.start, s.end)]


def locate_anchor(doc: WordsDoc, kept_spans: Sequence, phrase: str) -> list[tuple[int, int]]:
    """Where `phrase` occurs among the words that survive the cut.

    Matching runs on normalised tokens (`textnorm`) because the model retypes
    Darija with different alef and taa marbuta forms even when it is copying.
    Returns `(first_index, last_index_exclusive)` spans in SOURCE word indices.
    """
    needle = textnorm.normalize_tokens(phrase)
    if not needle:
        return []
    kept = _kept_indices(kept_spans)
    haystack = [doc.words[i].norm for i in kept]
    hits: list[tuple[int, int]] = []
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            hits.append((kept[i], kept[i + len(needle) - 1] + 1))
    return hits


def _validate_inserts(doc: WordsDoc, kept_spans: Sequence,
                      raw_inserts: Sequence[dict]) -> tuple[list[Insert], list[str]]:
    inserts: list[Insert] = []
    failures: list[str] = []
    for n, item in enumerate(raw_inserts or []):
        after = (item.get("after_text") or "").strip()
        if not after:
            failures.append(f"insert {n}: after_text is empty")
            continue
        if len(textnorm.normalize_tokens(after)) < MIN_ANCHOR_TOKENS:
            failures.append(
                f"insert {n}: after_text {after!r} is too short to anchor; "
                f"copy 3 to 6 consecutive words from the transcript")
            continue
        hits = locate_anchor(doc, kept_spans, after)
        if not hits:
            failures.append(
                f"insert {n}: after_text {after!r} is not in the text you kept. "
                f"Copy it verbatim from a line you did not delete.")
            continue
        start, end = hits[0]
        inserts.append(Insert(
            after_text=after,
            claim=(item.get("claim") or "").strip(),
            entity=(item.get("entity") or "").strip(),
            prefer_source=(item.get("prefer_source") or None),
            # The visual appears where the anchor phrase starts; stage 5 backs
            # off by the profile's lead_in. Index, not seconds.
            trigger_word_index=start,
            trigger_word=doc.words[start].display or doc.words[start].word,
            anchor_span=(start, end),
            ambiguous=len(hits) > 1,
        ))
    return inserts, failures


def validate(doc: WordsDoc, payload: dict, profile: dict, source_name: str
             ) -> tuple[EditPlan | None, list[str]]:
    """Turn a model payload into a plan, or into the list of reasons it is not one."""
    failures: list[str] = []
    kept_text = (payload.get("kept_text") or "").strip() if isinstance(payload, dict) else ""
    if not kept_text:
        return None, ["the answer has no kept_text"]
    strategy = (payload.get("strategy") or "").strip()
    if not strategy:
        failures.append("strategy is missing: say in 4 to 8 sentences what you did and why")

    diff = diff_align.diff_text_against_words(doc, kept_text)
    if diff.invented:
        shown = ", ".join(repr(t) for t in diff.invented[:10])
        return None, [
            f"kept_text contains {len(diff.invented)} word(s) that are not in the raw "
            f"take: {shown}. You paraphrased. Delete material instead of rewriting it, "
            f"and copy every kept word exactly."]

    # The band is judged on the real thing: the same engine that will produce
    # the EDL, after padding and after short removals are merged away.
    try:
        cut_plan = derive_cuts.derive(doc, kept_text, profile, source_name)
    except ValueError as exc:
        # Nothing matched, or everything was cut: recoverable by the model, so
        # it is a rejection with a reason rather than a crash.
        return None, [f"the edit cannot be derived from the take: {exc}"]
    if not cut_plan.within_band:
        failures.append(
            f"cut_ratio is {cut_plan.cut_ratio:.2f} but the style profile wants "
            f"{cut_plan.target_cut_ratio} +/-{cut_plan.tolerance:.2f}. "
            + ("Keep more material." if cut_plan.cut_ratio > (cut_plan.target_cut_ratio or 0)
               else "Cut more aggressively."))

    inserts, insert_failures = _validate_inserts(doc, cut_plan.kept_spans,
                                                 payload.get("inserts") or [])
    failures += insert_failures
    if failures:
        return None, failures

    warnings = [f"insert {i}: {ins.after_text!r} occurs more than once; used the first"
                for i, ins in enumerate(inserts) if ins.ambiguous]
    warnings += _density_warnings(inserts, cut_plan.kept_duration)

    return EditPlan(
        source_name=source_name, kept_text=kept_text, strategy=strategy, inserts=inserts,
        cut_ratio=cut_plan.cut_ratio, target_cut_ratio=cut_plan.target_cut_ratio,
        within_band=cut_plan.within_band, warnings=warnings,
        needs_visual_check=cut_plan.needs_visual_check,
    ), []


def _density_warnings(inserts: Sequence[Insert], kept_duration: float) -> list[str]:
    """Too many visuals is a style problem, not a correctness one: warn, do not fail."""
    if kept_duration <= 0 or not inserts:
        return []
    try:
        ceiling = float(config.get(config.load("layout"),
                                   "inserts.density.max_per_minute", 4.0))
    except (FileNotFoundError, OSError):
        return []
    per_minute = len(inserts) / (kept_duration / 60.0)
    if per_minute > ceiling:
        return [f"{per_minute:.1f} inserts per minute exceeds the {ceiling:.1f} ceiling "
                f"in configs/layout.yaml"]
    return []


# -------- the planner ---------------------------------------------------------


def plan(doc: WordsDoc, packed_text: str, profile: dict, *, source_name: str,
         llm: gemini_client.LLM | None = None, few_shot: str = "",
         max_attempts: int = 2) -> EditPlan:
    """Ask the model for an edit, validate it, retry once with the reason, then fail."""
    llm = llm or gemini_client.default_llm(profile.get("llm"))
    feedback: str | None = None
    last: list[str] = []
    for attempt in range(1, max_attempts + 1):
        payload = llm.generate(
            build_prompt(packed_text, profile, few_shot, feedback),
            system=SYSTEM_PROMPT, schema=RESPONSE_SCHEMA, temperature=0.2,
            # A retry that reads its own cached answer is not a retry.
            use_cache=(attempt == 1),
        )
        if isinstance(payload, str):
            payload = json.loads(payload)
        result, failures = validate(doc, payload, profile, source_name)
        if result is not None:
            result.attempts = attempt
            return result
        last = failures
        feedback = "\n".join(f"- {f}" for f in failures)
    raise PlanError(
        f"the planner failed validation twice for {source_name}:\n"
        + "\n".join(f"  - {f}" for f in last), last)


def confirm(edit_plan: EditPlan, *, auto: bool = False,
            input_fn: Callable[[str], str] = input, out=None) -> bool:
    """Hard rule 10: the strategy is read by a human before anything is rendered."""
    # Resolved per call, not bound as a default: the caller may have replaced
    # sys.stdout (a wrapper CLI, a test) after this module was imported.
    out = out if out is not None else sys.stdout
    print("\n=== strategy " + "=" * 55, file=out)
    print(edit_plan.strategy, file=out)
    print(f"\ncut_ratio {edit_plan.cut_ratio:.2f} (target {edit_plan.target_cut_ratio}), "
          f"{len(edit_plan.inserts)} insert(s), "
          f"{len(edit_plan.needs_visual_check)} cut(s) to eyeball", file=out)
    for i in edit_plan.inserts:
        print(f"  insert @ word {i.trigger_word_index} ({i.trigger_word}): {i.claim}", file=out)
    for w in edit_plan.warnings:
        print(f"  warning: {w}", file=out)
    if auto:
        print("--auto: proceeding without confirmation; the report is the review.", file=out)
        edit_plan.confirmed = False
        return True
    if not sys.stdin.isatty() and input_fn is input:
        raise PlanError("no terminal to confirm on: pass --auto to run unattended")
    answer = input_fn("proceed with this edit? [y/N] ").strip().lower()
    edit_plan.confirmed = answer in ("y", "yes")
    return edit_plan.confirmed


# -------- cli -----------------------------------------------------------------


def _read_few_shot(profile: dict, profile_path: Path | None) -> str:
    """`few_shot_examples.md` lives next to the profile that names it."""
    name = profile.get("examples")
    if not name or profile_path is None:
        return ""
    p = Path(name)
    if not p.is_absolute():
        p = profile_path.parent / p
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="plan_edit",
        description="Stage 4: ask the editor model for kept_text + insert markers.")
    ap.add_argument("--words", required=True, help="path to <name>.words.json")
    ap.add_argument("--profile", help="style_profile.json from `zeta learn`")
    ap.add_argument("--packed", help="takes_packed.md (default: derived from words.json)")
    ap.add_argument("--videos-dir", help="output root (default: the source video's folder)")
    ap.add_argument("--source-name", help="EDL source key")
    ap.add_argument("--auto", action="store_true", help="skip the strategy confirmation")
    args = ap.parse_args(argv)

    doc = WordsDoc.load(args.words)
    profile_path = Path(args.profile).resolve() if args.profile else None
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path else {}
    name = args.source_name or doc.source.get("name") or Path(args.words).stem.split(".")[0]

    packed = (Path(args.packed).read_text(encoding="utf-8") if args.packed
              else packed_view(doc))
    videos_dir = args.videos_dir or doc.source.get("path") or Path(args.words).parent
    out = paths.EditPaths.for_videos_dir(videos_dir).ensure()

    edit_plan = plan(doc, packed, profile, source_name=name,
                     few_shot=_read_few_shot(profile, profile_path))
    edit_plan.words_path = str(Path(args.words).resolve())
    edit_plan.profile_path = str(profile_path) if profile_path else None

    if not confirm(edit_plan, auto=args.auto):
        print("aborted: nothing written")
        return 1
    print(f"wrote {edit_plan.save(out.plan)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
