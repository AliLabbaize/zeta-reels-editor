"""Where everything lives.

Hard rule 11: every artifact this tool produces goes under `<videos_dir>/edit/`.
Nothing is ever written into the repo working tree at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True)
class EditPaths:
    """Resolved output layout for one editing session."""

    videos_dir: Path
    edit: Path

    @classmethod
    def for_videos_dir(cls, videos_dir: str | Path) -> "EditPaths":
        v = Path(videos_dir).resolve()
        if v.is_file():
            v = v.parent
        return cls(videos_dir=v, edit=v / "edit")

    # -- subdirectories -----------------------------------------------------
    @property
    def transcripts(self) -> Path:
        return self.edit / "transcripts"

    @property
    def audio(self) -> Path:
        return self.edit / "audio"

    @property
    def captions(self) -> Path:
        return self.edit / "captions"

    @property
    def screenshots(self) -> Path:
        return self.edit / "screenshots"

    @property
    def clips(self) -> Path:
        return self.edit / "clips_graded"

    @property
    def verify(self) -> Path:
        return self.edit / "verify"

    @property
    def learn(self) -> Path:
        return self.edit / "learn"

    # -- files --------------------------------------------------------------
    @property
    def edl(self) -> Path:
        return self.edit / "edl.json"

    @property
    def packed(self) -> Path:
        return self.edit / "takes_packed.md"

    @property
    def plan(self) -> Path:
        return self.edit / "edit_plan.json"

    @property
    def report(self) -> Path:
        return self.edit / "decision_report.html"

    @property
    def project_md(self) -> Path:
        return self.edit / "project.md"

    @property
    def final(self) -> Path:
        return self.edit / "final.mp4"

    @property
    def preview(self) -> Path:
        return self.edit / "preview.mp4"

    def slot(self, slot_id: str) -> Path:
        return self.screenshots / slot_id

    def words_json(self, source_name: str) -> Path:
        return self.transcripts / f"{source_name}.words.json"

    def ensure(self) -> "EditPaths":
        for d in (
            self.edit, self.transcripts, self.audio, self.captions,
            self.screenshots, self.verify,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self
