import logging
import argparse
from datetime import datetime, timedelta
from orbit_predictor.sources import NoradTLESource
from . import apt, georef


def run():
    parser = argparse.ArgumentParser(
        prog="bluedot",
        description="decode and georeference images from weather satellites",
    )
    parser.add_argument("input")
    parser.add_argument("-t", "--time")
    parser.add_argument("-s", "--satellite")
    parser.add_argument("-d", "--tle-file")
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)

    if args.time and args.tle_file and args.satellite:
        time = datetime.fromisoformat(args.time)
        tle_source = NoradTLESource.from_file(args.tle_file)
        satellite = tle_source.get_predictor(args.satellite)
        do_georef = True
    else:
        logging.warn(
            "Resulting image will not be georeferenced: "
            "arguments --time, --satellite, or --tle-file are missing"
        )
        do_georef = False

    logging.info("Loading file")
    rate, signal = apt.read_signal(args.input)
    images = apt.apt_decode(rate, signal)

    base_path = args.input.removesuffix(".wav")
    suffixes = ("_a.tif", "_b.tif")
    logging.info(f"Saving images to {base_path}_*.tif")

    for image, suffix in zip(images, suffixes):
        image_path = base_path + suffix
        image.save(image_path)

        if do_georef:
            duration = timedelta(seconds=len(signal) / rate)
            georef.compute(image_path, satellite, time, time + duration, apt.span)
