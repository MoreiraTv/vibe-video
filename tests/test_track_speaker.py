import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import sys
import tempfile

# Ensure the helpers module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.track_speaker import apply_ema, track_stage

class TestTrackSpeaker(unittest.TestCase):
    def test_apply_ema(self):
        """Test the Exponential Moving Average smoothing logic"""
        self.assertEqual(apply_ema([]), [])
        # Constant values shouldn't change
        self.assertEqual(apply_ema([10, 10, 10], alpha=0.5), [10, 10.0, 10.0])
        # Step change should smooth towards the new value
        self.assertEqual(apply_ema([0, 10, 10], alpha=0.5), [0, 5.0, 7.5])
        
    @patch("helpers.track_speaker.YOLO")
    @patch("helpers.track_speaker.cv2.VideoCapture")
    def test_track_stage(self, mock_video_capture, mock_yolo):
        """Test the tracking logic utilizing mocked YOLO predictions"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        import cv2
        def mock_get(prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 3
            return 0
            
        mock_cap.get.side_effect = mock_get
        mock_video_capture.return_value = mock_cap
        
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        # --- Mock YOLO Tracking Output ---
        
        class MockTensor:
            def __init__(self, vals):
                self.vals = vals
            def tolist(self):
                return self.vals

        # Frame 1: Initial detection.
        box1 = MagicMock()
        box1.id = [1]
        box1.xyxy = [MockTensor([10, 10, 20, 20])] # area=100, center=(15, 15)
        
        box2 = MagicMock()
        box2.id = [2]
        box2.xyxy = [MockTensor([10, 10, 50, 50])] # area=1600, center=(30, 30)
        
        boxes_mock_f1 = MagicMock()
        boxes_mock_f1.__iter__.return_value = [box1, box2]
        boxes_mock_f1.id = [1, 2]
        
        res1 = MagicMock()
        res1.boxes = boxes_mock_f1
        
        # Frame 2: Person 2 moves slightly.
        box2_f2 = MagicMock()
        box2_f2.id = [2]
        box2_f2.xyxy = [MockTensor([20, 20, 60, 60])] # center=(40, 40)
        
        boxes_mock_f2 = MagicMock()
        boxes_mock_f2.__iter__.return_value = [box2_f2]
        boxes_mock_f2.id = [2]
        
        res2 = MagicMock()
        res2.boxes = boxes_mock_f2
        
        # Frame 3: Target disappears from detection. 
        # The script should retain the previous known center (40, 40).
        boxes_mock_f3 = MagicMock()
        boxes_mock_f3.id = None
        
        res3 = MagicMock()
        res3.boxes = boxes_mock_f3
        
        # Yield our simulated frames in sequence
        mock_model.track.return_value = [res1, res2, res3]
        
        # Execute test within a temporary directory to avoid leaving artifact files
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_json = Path(tmp_dir) / "output.json"
            
            # alpha=1.0 completely overrides the EMA, making it easier to assert raw values
            track_stage(Path("fake_video.mp4"), output_json, alpha=1.0)
            
            self.assertTrue(output_json.exists())
            with open(output_json, "r") as f:
                data = json.load(f)
                
            self.assertEqual(data["mode"], "stage")
            self.assertEqual(data["fps"], 30.0)
            self.assertEqual(data["total_frames"], 3)
            self.assertEqual(data["target_id"], 2)
            
            frames = data["frames"]
            self.assertEqual(len(frames), 3)
            
            # Verify coordinates follow the mocked detections
            self.assertEqual(frames[0]["cx"], 30.0)
            self.assertEqual(frames[0]["cy"], 30.0)
            
            self.assertEqual(frames[1]["cx"], 40.0)
            self.assertEqual(frames[1]["cy"], 40.0)
            
            self.assertEqual(frames[2]["cx"], 40.0)
            self.assertEqual(frames[2]["cy"], 40.0)

if __name__ == '__main__':
    unittest.main()
