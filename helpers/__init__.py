"""Zeta auto editor helpers.

Import-cheap by design: every module in this package must import with only the
standard library plus PyYAML. Heavy or optional dependencies (google-genai,
whisperx, torch, cv2, scenedetect, matplotlib, PIL) are imported lazily inside
the function that needs them, so `zeta plan` never pays for torch and the test
suite runs on a bare interpreter.
"""

__version__ = "0.1.0"
