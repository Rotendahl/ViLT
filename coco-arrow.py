import sys
from vilt.utils.write_coco_karpathy import make_arrow
root = "/science/image/nlp-datasets/emanuele/data/mscoco/"
arrows_root = "/home/xmt224/erda/vilt-arrow/mscoco/"
make_arrow(sys.argv[1], sys.argv[2])
