import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from doctor import run_decision_doctor, run_image_doctor
from resource_plan import CgroupError


class DoctorCgroupTest(unittest.TestCase):
    def test_unreadable_cgroup_is_reported_as_a_memory_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            header = json.dumps({"w": {"dtype": "F32", "shape": [1],
                                       "data_offsets": [0, 4]}}).encode()
            (model / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + bytes(4))
            for doctor in (run_image_doctor, run_decision_doctor):
                with self.subTest(doctor=doctor.__name__), \
                        patch("doctor.memory_available", side_effect=CgroupError("unreadable cgroup")):
                    report = doctor(model, model / "engine")
                    memory = next(check for check in report["checks"] if check["id"] == "memory.ram")
                    self.assertEqual(memory["status"], "fail")
                    self.assertIn("unreadable cgroup", memory["summary"])
                    self.assertFalse(any(check["id"] == "model.weights" and check["status"] == "fail"
                                         for check in report["checks"]))


if __name__ == "__main__":
    unittest.main()
