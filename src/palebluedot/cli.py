import sys
import logging
from . import apt


def run():
    if len(sys.argv) <= 1:
        print(f"Usage: bluedot [input file]")
        sys.exit(1)

    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    input_path = sys.argv[1]
    base_path = input_path.removesuffix(".wav")

    logging.info("Loading file")
    rate, signal = apt.read_signal(input_path)
    image_a, image_b = apt.apt_decode(rate, signal)

    logging.info(f"Saving images to {base_path}_*.tif")
    image_a.save(base_path + "_a.tif")
    image_b.save(base_path + "_b.tif")
