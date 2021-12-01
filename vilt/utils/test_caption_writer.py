import unittest
import os

import write_cc12 as caption_writer

class TestCocoWriter(unittest.TestCase):
	def setUp(self) -> None:
		self.root_path = '/Users/rotendahl/Documents/KU/Courses/thesis/code/ViLT/assets/test_12gcc'
		self.dest_path = '/Users/rotendahl/Documents/KU/Courses/thesis/code/ViLT/assets/test_12gcc/output'

	def test_load_captions(self):
		self.assertEqual(
			len(caption_writer.load_captions(self.root_path, 'train')),
			900
		)
		self.assertEqual(
			len(caption_writer.load_captions(self.root_path, 'val')),
			100
		)

	def test_get_image_paths(self):
		self.assertEqual(
			len(caption_writer.get_image_paths(self.root_path, 'train')),
			900
		)
		self.assertEqual(
			len(caption_writer.get_image_paths(self.root_path, 'val')),
			100
		)

	def test_filter_images(self):
		captions = caption_writer.load_captions(self.root_path, 'train')
		image_paths = caption_writer.get_image_paths(self.root_path, 'train')
		self.assertEqual(
			len(caption_writer.filter_images(captions, image_paths)),
			len(captions)
		)
		self.assertEqual(
			len(caption_writer.filter_images(captions, image_paths + ["hello/hello"])),
			len(captions)
		)

	def test_create_work_batches_indices(self):
		train_image_paths = caption_writer.get_image_paths(self.root_path, 'train')
		val_image_paths = caption_writer.get_image_paths(self.root_path, 'val')

		train_batches = caption_writer.create_work_batches_indices(train_image_paths, 100)
		val_batches = caption_writer.create_work_batches_indices(val_image_paths, 100)
		self.assertEqual(
			len(train_batches),
			9
		)
		[self.assertEqual(len(train_image_paths[a:b]), 100) for (a,b) in train_batches]

		# Check that all images are in a batch
		all_train_batches = []
		for (a,b) in train_batches:
			all_train_batches += train_image_paths[a:b]
		self.assertEqual(train_image_paths, all_train_batches)

		self.assertEqual(
			len(val_batches),
			1
		)
		(val_i, val_j) = val_batches[0]
		self.assertEqual(
			val_image_paths[val_i:val_j],
			val_image_paths
		)

	def test_make_arrow(self):
		caption_writer.make_arrow(self.root_path, self.dest_path, 10)


if __name__ == '__main__':
    unittest.main()