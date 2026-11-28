"""
stroop_task.py – Stroop color-word interference test (High load validation).

Shows a color word (e.g. "RED") printed in an incongruent ink color
(e.g. blue). Participant must press the key for the INK COLOR, not the word.

Keys: R=Red  G=Green  B=Blue  Y=Yellow  |  ESC=quit

Saves trial-by-trial results to data/sessions/<session_id>_stroop.csv
"""

from __future__ import annotations

import csv
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"

# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------
COLORS = {
    "RED":    (0,   0,   220),   # BGR
    "GREEN":  (0,   180,  0),
    "BLUE":   (220, 0,    0),
    "YELLOW": (0,   220, 220),
}
COLOR_KEYS = {"r": "RED", "g": "GREEN", "b": "BLUE", "y": "YELLOW"}

TRIAL_SECS   = 3.0
TOTAL_TRIALS = 48    # 12 congruent + 36 incongruent


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class StroopTrial:
    idx:        int
    word:       str
    ink:        str
    congruent:  bool
    response:   str    # color name or "" if no response
    correct:    bool
    rt_ms:      float


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class StroopTask:
    def __init__(self, session_id: str = "test", total: int = TOTAL_TRIALS):
        self.session_id = session_id
        self.total      = total
        self.trials: list[StroopTrial] = []
        self.results: list[StroopTrial] = []
        self._generate()

    def _generate(self) -> None:
        words  = list(COLORS.keys())
        trials = []
        for i in range(self.total):
            congruent = i < self.total // 4
            word      = random.choice(words)
            if congruent:
                ink = word
            else:
                ink = random.choice([c for c in words if c != word])
            trials.append((word, ink, congruent))
        random.shuffle(trials)
        self.trials = trials   # type: ignore

    def run(self) -> list[StroopTrial]:
        win = "Stroop Task  (press key for INK COLOR: R G B Y | ESC=quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 500)

        for idx, (word, ink, congruent) in enumerate(self.trials):
            ink_color = COLORS[ink]
            frame_start = time.monotonic()
            response = ""
            rt_ms    = 0.0

            deadline = frame_start + TRIAL_SECS
            img = self._make_frame(word, ink_color, idx)
            cv2.imshow(win, img)

            while time.monotonic() < deadline:
                key = cv2.waitKey(10) & 0xFF
                if key == 27:
                    cv2.destroyAllWindows()
                    self._save()
                    return self.results
                ch = chr(key).lower() if 0 < key < 128 else ""
                if ch in COLOR_KEYS and not response:
                    response = COLOR_KEYS[ch]
                    rt_ms = (time.monotonic() - frame_start) * 1000

            correct = response == ink
            trial = StroopTrial(
                idx=idx, word=word, ink=ink, congruent=congruent,
                response=response, correct=correct, rt_ms=rt_ms,
            )
            self.results.append(trial)

            blank = np.zeros((500, 800, 3), dtype=np.uint8)
            cv2.imshow(win, blank)
            cv2.waitKey(250)

        cv2.destroyAllWindows()
        self._save()
        return self.results

    def _make_frame(self, word: str, ink_color: tuple, idx: int) -> np.ndarray:
        img = np.zeros((500, 800, 3), dtype=np.uint8)
        prog = int(800 * idx / max(len(self.trials), 1))
        cv2.rectangle(img, (0, 0), (prog, 8), (80, 80, 180), -1)

        (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 6)
        cx = (800 - tw) // 2
        cy = (500 + th) // 2
        cv2.putText(img, word, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 3.5, ink_color, 6)

        cv2.putText(img, "Press key for the INK COLOR:  R=Red  G=Green  B=Blue  Y=Yellow",
                    (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(img, f"Trial {idx + 1}/{len(self.trials)}",
                    (650, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        return img

    def _save(self) -> None:
        if not self.results:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / f"{self.session_id}_stroop.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.results[0]).keys()))
            writer.writeheader()
            writer.writerows([asdict(r) for r in self.results])
        print(f"Stroop results saved → {path}")

    def score(self) -> dict:
        inc = [r for r in self.results if not r.congruent]
        con = [r for r in self.results if r.congruent]
        return {
            "total": len(self.results),
            "congruent_acc":   sum(r.correct for r in con) / max(len(con), 1),
            "incongruent_acc": sum(r.correct for r in inc) / max(len(inc), 1),
            "stroop_effect_rt": (
                float(np.mean([r.rt_ms for r in inc if r.response])) -
                float(np.mean([r.rt_ms for r in con if r.response]))
            ) if inc and con else 0.0,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    task = StroopTask(session_id=sid)
    print(f"Stroop task – {len(task.trials)} trials – session: {sid}")
    print("Press the KEY matching the INK COLOR (not the word).")
    input("Press ENTER to begin...")
    task.run()
    print("\n--- Score ---")
    for k, v in task.score().items():
        print(f"  {k}: {v}")
