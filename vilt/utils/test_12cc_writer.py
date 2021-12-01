import unittest
import os

import write_cc12 as caption_writer


class TestCocoWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.root_path = (
            "/Users/rotendahl/Documents/KU/Courses/thesis/code/ViLT/assets/test_12gcc"
        )
        self.dest_path = "/Users/rotendahl/Documents/KU/Courses/thesis/code/ViLT/assets/test_12gcc/output"

    def test_load_captions(self):
        self.assertEqual(
            len(caption_writer.load_captions(self.root_path, "train")), 1000
        )
        self.assertEqual(len(caption_writer.load_captions(self.root_path, "val")), 100)

    def test_get_image_paths(self):
        self.assertEqual(len(caption_writer.get_image_paths(self.root_path)), 1000)

    def test_filter_images(self):
        captions = caption_writer.load_captions(self.root_path, "train")
        image_paths = caption_writer.get_image_paths(self.root_path)
        self.assertEqual(
            len(caption_writer.filter_images(captions, image_paths)), len(captions)
        )
        self.assertEqual(
            len(caption_writer.filter_images(captions, image_paths + ["hello/hello"])),
            len(captions),
        )

    def test_create_work_batches_indices(self):
        train_captions = caption_writer.load_captions(self.root_path, "train")
        val_captions = caption_writer.load_captions(self.root_path, "val")

        train_image_paths = caption_writer.get_image_paths(self.root_path)
        val_image_paths = caption_writer.get_image_paths(self.root_path)

        train_image_paths = caption_writer.filter_images(
            train_captions, train_image_paths
        )
        val_image_paths = caption_writer.filter_images(val_captions, val_image_paths)

        train_batches = caption_writer.create_work_batches_indices(train_image_paths)
        val_batches = caption_writer.create_work_batches_indices(val_image_paths)
        self.assertEqual(len(train_batches), 10)
        [self.assertEqual(len(batch), 100) for batch in train_batches]

        # Check that all images are in a batch
        all_train_batches = {}
        for batch in train_batches:
            for b_id in batch:
                all_train_batches[b_id] = train_image_paths[b_id]
        self.assertEqual(train_image_paths, all_train_batches)

        self.assertEqual(len(val_batches), 1)
        all_val_batches = {}
        for batch in val_batches:
            for b_id in batch:
                all_val_batches[b_id] = val_image_paths[b_id]

        self.assertEqual(all_val_batches, val_image_paths)

    def test_make_arrow(self):
        caption_writer.make_arrow(self.root_path, self.root_path, self.dest_path)


if __name__ == "__main__":
    unittest.main()