"""
n_back_task.py – 3-back working memory task (High cognitive load stimulus).

Presents a sequence of single letters. Participant must press SPACE
when the current letter matches the one shown 3 steps ago (n=3).

Logs: response times, hit/miss/false-alarm counts.
Saves results to data/sessions/<session_id>_nback.csv

Controls: SPACE = match response | ESC = quit
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N          = 3            # n-back level
TRIAL_SECS = 2.5          # seconds per stimulus
TOTAL_TRIALS = 60         # ~2.5 min + practice
TARGET_PROB  = 0.30       # probability that trial is a target

LETTERS = list("BCDFGHJKLMNPQRSTVWXYZ")  # avoid ambiguous letters A, E, I, O, U

DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
    trial_idx:   int
    letter:      str
    is_target:   bool
    responded:   bool
    correct:     bool
    rt_ms:       float = 0.0  # reaction time (ms); 0 if no response


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class NBackTask:
    def __init__(self, session_id: str = "test", n: int = N, total_trials: int = TOTAL_TRIALS):
        self.session_id  = session_id
        self.n           = n
        self.total_trials = total_trials
        self.sequence: list[str] = []
        self.results: list[TrialResult] = []
        self._generate_sequence()

    def _generate_sequence(self) -> None:
        seq = [random.choice(LETTERS) for _ in range(self.n)]
        for _ in range(self.total_trials - self.n):
            if random.random() < TARGET_PROB and len(seq) >= self.n:
                seq.append(seq[-self.n])  # target: match n steps back
            else:
                candidates = [c for c in LETTERS if c != seq[-self.n]]
                seq.append(random.choice(candidates))
        self.sequence = seq

    def run(self) -> list[TrialResult]:
        """Run the task with an OpenCV window. Returns trial results."""
        win = "N-Back Task  (SPACE = match | ESC = quit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 800, 500)

        for idx, letter in enumerate(self.sequence):
            is_target = idx >= self.n and letter == self.sequence[idx - self.n]
            frame_start = time.monotonic()
            responded = False
            rt_ms = 0.0

            deadline = frame_start + TRIAL_SECS

            # Draw frame
            img = self._make_frame(letter, idx, len(self.sequence))
            cv2.imshow(win, img)

            # Collect response within trial window
            while time.monotonic() < deadline:
                key = cv2.waitKey(10) & 0xFF
                if key == 27:          # ESC
                    cv2.destroyAllWindows()
                    self._save()
                    return self.results
                if key == ord(" ") and not responded:
                    responded = True
                    rt_ms = (time.monotonic() - frame_start) * 1000

            correct = responded == is_target
            self.results.append(TrialResult(
                trial_idx=idx,
                letter=letter,
                is_target=is_target,
                responded=responded,
                correct=correct,
                rt_ms=rt_ms,
            ))

            # Brief blank between trials
            blank = np.zeros((500, 800, 3), dtype=np.uint8)
            cv2.imshow(win, blank)
            cv2.waitKey(300)

        cv2.destroyAllWindows()
        self._save()
        return self.results

    def _make_frame(self, letter: str, idx: int, total: int) -> np.ndarray:
        img = np.zeros((500, 800, 3), dtype=np.uint8)
        # Progress bar
        prog = int(800 * idx / total)
        cv2.rectangle(img, (0, 0), (prog, 8), (80, 180, 80), -1)
        # Letter
        (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 8, 10)
        cx, cy = (800 - tw) // 2, (500 + th) // 2
        cv2.putText(img, letter, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 255, 255), 10)
        # Instructions
        cv2.putText(img, f"Press SPACE if this letter appeared {self.n} steps ago",
                    (40, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(img, f"Trial {idx + 1}/{total}",
                    (640, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return img

    def _save(self) -> None:
        if not self.results:
            print("No results to save.")
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / f"{self.session_id}_nback.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.results[0]).keys()))
            writer.writeheader()
            writer.writerows([asdict(r) for r in self.results])
        print(f"N-back results saved → {path}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self) -> dict:
        targets     = [r for r in self.results if r.is_target]
        non_targets = [r for r in self.results if not r.is_target]
        hits        = sum(1 for r in targets if r.responded)
        fa          = sum(1 for r in non_targets if r.responded)
        misses      = len(targets) - hits
        rts         = [r.rt_ms for r in targets if r.responded]
        return {
            "n": self.n,
            "total_trials": len(self.results),
            "hits": hits,
            "misses": misses,
            "false_alarms": fa,
            "accuracy": hits / max(len(targets), 1),
            "mean_rt_ms": float(np.mean(rts)) if rts else 0.0,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    task = NBackTask(session_id=sid)
    print(f"Starting {N}-back task  ({TOTAL_TRIALS} trials)  –  session: {sid}")
    print("Press SPACE when the current letter matches the one shown 3 steps ago.")
    input("Press ENTER to begin...")
    results = task.run()
    print("\n--- Score ---")
    for k, v in task.score().items():
        print(f"  {k}: {v}")
