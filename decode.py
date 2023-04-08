#!/usr/bin/env python3
import numpy as np
from PIL import Image
import scipy.signal
import scipy.io
import sys
import logging


line_width = 1040
lines_per_second = 4
sync_a_pattern = "000011001100110011001100110011000000000"
sync_b_pattern = "000011100111001110011100111001110011100"


def read_signal(path):
    rate, signal = scipy.io.wavfile.read(path)
    return rate, signal.mean(axis=1)


def amplitude_demod(signal):
    envelope = np.abs(scipy.signal.hilbert(signal))
    return (envelope * 256 / 6000).clip(0, 255).astype("int")


def gen_sync_signal(pattern, samples_per_symbol):
    length = int(samples_per_symbol * len(pattern))
    sampled_pattern = np.array([
        int((pattern + "0")[round(i / samples_per_symbol)]) - 0.5
        for i in range(length)
    ])
    return sampled_pattern


def find_syncs(levels, sync_signal, samples_per_symbol):
    corr = np.correlate(levels - 128, sync_signal)
    corr = corr / np.max(corr)
    peaks, _ = scipy.signal.find_peaks(
        corr,
        height=.75,
        distance=samples_per_symbol * line_width,
    )
    return peaks


def image_from_signal(levels, syncs, samples_per_symbol):
    first_sync = syncs[0]
    last_sync = syncs[-1]

    line_width_samples = round(line_width * samples_per_symbol)
    sync_dist = round(line_width * samples_per_symbol * 2)

    lines = int((last_sync - first_sync) / sync_dist)
    data = np.zeros((lines + 1, line_width))

    for sync in syncs:
        y = int((sync - first_sync) / sync_dist)
        line = levels[sync : sync + line_width_samples]
        data[y] = scipy.signal.resample(line, num=line_width)

    return Image.fromarray(data.astype("uint8"))


def decode(input_path):
    rate, signal = read_signal(input_path)
    samples_per_symbol = rate / (line_width * lines_per_second)

    logging.info("Demodulating signal")
    levels = amplitude_demod(signal)

    logging.info("Finding syncs for channel A")
    sync_a_signal = gen_sync_signal(sync_a_pattern, samples_per_symbol)
    syncs_a = find_syncs(levels, sync_a_signal, samples_per_symbol)
    logging.info("Decoding channel A")
    image_a = image_from_signal(levels, syncs_a, samples_per_symbol)

    logging.info("Finding syncs for channel B")
    sync_b_signal = gen_sync_signal(sync_b_pattern, samples_per_symbol)
    syncs_b = find_syncs(levels, sync_b_signal, samples_per_symbol)
    logging.info("Decoding channel B")
    image_b = image_from_signal(levels, syncs_b, samples_per_symbol)

    return image_a, image_b


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(f"Usage: {sys.argv[0]} [input file]")
        sys.exit(1)

    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    input_path = sys.argv[1]
    base_path = input_path.removesuffix(".wav")
    image_a, image_b = decode(input_path)

    logging.info(f"Saving images to {base_path}_*.png")
    image_a.save(base_path + "_a.png")
    image_b.save(base_path + "_b.png")
