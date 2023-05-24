import numpy as np
from PIL import Image
import scipy.signal
import scipy.io
import logging
from .xcorr import correlate_template


# Number of symbols in a complete APT line
line_width = 1040

# Number of APT lines transmitted every second
lines_per_second = 4

# Carrier frequency used to transmit the signal
carrier_frequency = 2400

# Synchronization pattern for channel A
sync_a_pattern = "000011001100110011001100110011000000000"

# Synchronization pattern for channel B
sync_b_pattern = "000011100111001110011100111001110011100"


def read_signal(path):
    """
    Read a signal from a WAV file.

    :param path: path to the file.
    :return: sampling rate and list of samples.
    """
    rate, signal = scipy.io.wavfile.read(path)

    if len(signal.shape) == 2:
        # Merge all channels into one
        signal = signal.mean(axis=1)

    return rate, signal.astype("float")


def amplitude_demod(rate, signal):
    """
    Demodulate an amplitude-modulated signal.

    :param rate: signal sampling rate.
    :param signal: list of samples.
    :return: demodulated signal.
    """
    lowpass = scipy.signal.butter(
        N=5,
        Wn=carrier_frequency * 1.1,
        btype="lowpass",
        output="sos",
        fs=rate,
    )
    return (scipy.signal.sosfilt(lowpass, signal ** 2).clip(0) * 2) ** 0.5


def gen_sync_signal(pattern, samples_per_symbol):
    """
    Generate a square synchronization signal from a pattern.

    :param pattern: binary synchronization pattern.
    :param samples_per_symbol: number of samples for each sync symbol.
    :return: generated synchronization signal.
    """
    length = int(samples_per_symbol * len(pattern))
    return np.array([
        int((pattern + "0")[round(i / samples_per_symbol)])
        for i in range(length)
    ])


def find_syncs(levels, sync_signal, samples_per_symbol):
    """
    Look for synchronization patterns inside a signal.

    :param levels: demodulated signal.
    :param sync_signal: signal to look for.
    :param samples_per_symbol: number of samples for each sync symbol.
    :return: list of candidate synchronization times.
    """
    corr = correlate_template(levels, sync_signal)
    peaks, _ = scipy.signal.find_peaks(
        corr,
        height=.5,
        distance=samples_per_symbol * line_width,
    )
    return peaks


def image_from_signal(levels, syncs, samples_per_symbol):
    """
    Reconstruct an image from an APT signal.

    :param levels: demodulated signal.
    :param syncs: synchronization times.
    :param samples_per_symbol: number of samples for each sync symbol.
    :return: decoded image.
    """
    first_sync = syncs[0]
    last_sync = syncs[-1]

    line_width_samples = round(line_width * samples_per_symbol)
    sync_dist = round(line_width * samples_per_symbol * 2)

    lines = int((last_sync - first_sync) / sync_dist)
    data = np.zeros((lines + 1, line_width), dtype="float")

    for sync in syncs:
        y = int((sync - first_sync) / sync_dist)
        line = levels[sync : sync + line_width_samples]
        data[y] = scipy.signal.resample(line, num=line_width)

    return Image.fromarray((data / 6000 * 255).round().clip(0, 255).astype("uint8"))


def apt_decode(rate, signal):
    """
    Decode both image channels from an APT signal.

    :param rate: signal sampling rate.
    :param signal: list of samples.
    :return: two images, one for each channel.
    """
    samples_per_symbol = rate / (line_width * lines_per_second)

    logging.info("Demodulating signal")
    levels = amplitude_demod(rate, signal)

    channels = []

    for channel, pattern in (("A", sync_a_pattern), ("B", sync_b_pattern)):
        logging.info(f"Finding syncs for channel {channel}")
        sync_signal = gen_sync_signal(pattern, samples_per_symbol)
        syncs = find_syncs(levels, sync_signal, samples_per_symbol)

        logging.info(f"Decoding channel {channel}")
        channels.append(image_from_signal(levels, syncs, samples_per_symbol))

    return channels
