import sys
import unittest
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from browser_stream_state import BrowserStreamState
from window_buffer import WindowFeatures


class BrowserStreamStateTests(unittest.TestCase):
    def test_feature_snapshots_are_copied_and_sample_ids_increase(self) -> None:
        state = BrowserStreamState()
        state.begin()
        features = WindowFeatures(ear_mean_avg=0.3, n_frames=15)

        state.publish_features(features, timestamp=10.0, buffered_frames=20)
        features.ear_mean_avg = 0.0
        snapshot = state.snapshot()
        self.assertEqual(snapshot.sample_id, 1)
        self.assertEqual(snapshot.latest_features.ear_mean_avg, 0.3)

        snapshot.latest_features.ear_mean_avg = 0.1
        self.assertEqual(state.snapshot().latest_features.ear_mean_avg, 0.3)

    def test_frame_updates_are_thread_safe(self) -> None:
        state = BrowserStreamState()
        state.begin()

        def publish_frames() -> None:
            for _ in range(100):
                state.publish_frame(face_found=True, buffered_frames=20)

        workers = [Thread(target=publish_frames) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        snapshot = state.snapshot()
        self.assertEqual(snapshot.frames_received, 400)
        self.assertTrue(snapshot.face_found)

    def test_stop_keeps_estimates_but_marks_stream_inactive(self) -> None:
        state = BrowserStreamState()
        state.begin()
        state.publish_features(WindowFeatures(n_frames=15), 10.0, 20)
        state.stop()

        snapshot = state.snapshot()
        self.assertFalse(snapshot.running)
        self.assertEqual(snapshot.sample_id, 1)
        self.assertIsNotNone(snapshot.latest_features)


if __name__ == "__main__":
    unittest.main()
