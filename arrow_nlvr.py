from vilt.utils.write_nlvr2 import make_arrow
import sys

if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    make_arrow(in_dir, out_dir)
