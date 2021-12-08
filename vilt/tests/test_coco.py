import unittest
import os

import vilt.utils.write_coco_karpathy as caption_writer


class TestCocoWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.root_path = (
            "/Users/rotendahl/Documents/KU/Courses/thesis/code/ViLT/assets/test_coco"
        )
        self.dest_path = "/Users/rotendahl/Documents/KU/Courses/thesis/code/ViLT/assets/test_coco/output"

    def test_make_arrow(self):
        caption_writer.make_arrow(self.root_path, self.dest_path)


if __name__ == "__main__":
    unittest.main()