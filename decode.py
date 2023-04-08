import numpy as np
from PIL import Image
import scipy.signal
import scipy.io


line_width = 1040
lines_per_second = 4
sync_a_pattern = "000011001100110011001100110011000000000"
sync_b_pattern = "000011100111001110011100111001110011100"


def read_signal(path):
    rate, data = scipy.io.wavfile.read(path)
    return rate, data.mean(axis=1)


def amplitude_demod(signal):
    envelope = np.abs(scipy.signal.hilbert(signal))
    return (envelope * 256 / 6000).clip(0, 255).astype("int")


def gen_sync_signal(pattern, samples_per_symbol):
    length = int(samples_per_symbol * len(pattern))
    sampled_pattern = np.array([
        int((pattern + "0")[round(i / samples_per_symbol)])
        for i in range(length)
    ])
    return sampled_pattern * 256


def find_syncs(levels, sync_signal):
    corr = np.correlate(levels - 128, sync_signal - 128)
    corr = corr / np.max(corr) * 256
    peaks, _ = scipy.signal.find_peaks(
        corr,
        width=20,
        height=100,
        distance=20000,
    )
    return peaks


def decode_image(levels, syncs, samples_per_second):
    start = syncs[0]
    end = syncs[-1]

    samples_per_symbol = samples_per_second / (line_width * lines_per_second)
    sync_dist = samples_per_second / lines_per_second * 2

    lines = int((end - start) / sync_dist)
    data = np.zeros((lines + 1, line_width))

    for peak in syncs:
        y = int((peak - start) / sync_dist)
        line = levels[peak:peak + round(line_width * samples_per_symbol)]
        data[y] = scipy.signal.resample(line, num=line_width)

    return Image.fromarray(data.astype("uint8"))


samples_per_second, signal = read_signal("noaa15-2023-04-05-19h14.wav")
samples_per_symbol = samples_per_second / (line_width * lines_per_second)

levels = amplitude_demod(signal)

sync_a_signal = gen_sync_signal(sync_a_pattern, samples_per_symbol)
syncs_a = find_syncs(levels, sync_a_signal)
decode_image(levels, syncs_a, samples_per_second).save("output_a.png")

sync_b_signal = gen_sync_signal(sync_b_pattern, samples_per_symbol)
syncs_b = find_syncs(levels, sync_b_signal)
decode_image(levels, syncs_b, samples_per_second).save("output_b.png")
