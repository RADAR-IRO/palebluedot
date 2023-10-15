import math 
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

# Distance seen on either side of the satellite (in meters)
span = 1_400_000


def gen_sync_signal(pattern, samples_per_symbol):
    """
    Generate a square synchronization signal from a pattern.

    :param pattern: binary synchronization pattern.
    :param samples_per_symbol: number of samples for each sync symbol.
    :return: generated synchronization signal.
    """
    length = int(samples_per_symbol * len(pattern))
    return np.array([
        int((pattern + "0")[math.floor(i / samples_per_symbol)])
        for i in range(length)
    ])


# Synchronization pattern for start of line in channel A
sync_a_pattern = "000011001100110011001100110011000000000"

# Synchronization pattern for start of line in channel B
sync_b_pattern = "000011100111001110011100111001110011100"

# Synchronization pattern for start of frame in both channels
frame_pattern = gen_sync_signal("123456780", samples_per_symbol=8)

# Telemetry and metadata
telemetry_values = 16
telemetry_lines = 8
sync_width = len(sync_a_pattern) + 47
telemetry_width = 45
frame_size = telemetry_values * telemetry_lines


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


def data_from_signal(levels, pattern, samples_per_symbol):
    """
    Extract data from a channel of an APT signal.

    :param levels: demodulated signal.
    :param pattern: channel synchronization pattern.
    :param samples_per_symbol: number of samples for each sync symbol.
    :return: decoded channel data.
    """
    line_width_samples = round(line_width * samples_per_symbol)
    sync_dist = round(line_width * samples_per_symbol * 2)

    # Find sync times
    sync_signal = gen_sync_signal(pattern, samples_per_symbol)
    corr = correlate_template(levels, sync_signal)
    syncs, _ = scipy.signal.find_peaks(corr, height=.5, distance=line_width_samples)

    # Extract a data line for each sync
    first_sync = syncs[0]
    last_sync = syncs[-1]
    lines = int((last_sync - first_sync) / sync_dist)

    data = np.zeros((lines + 1, line_width), dtype="float")

    for sync in syncs:
        y = int((sync - first_sync) / sync_dist)
        line = levels[sync : sync + line_width_samples]
        data[y] = scipy.signal.resample(line, num=line_width)

    return data


def rescale_data(data, low, high):
    return ((data - low) / (high - low) * 255).round().clip(0, 255)


def read_telemetry(data):
    """
    Read frame telemetry and produce an image from a decoded APT signal.

    :param data: channel data.
    :returns: image data and ID of the channel in use in the image.
    """
    # Align to frame starts
    telemetry_data = data[:, -telemetry_width:].mean(axis=1)
    corr = correlate_template(telemetry_data, frame_pattern)
    frame_starts, _ = scipy.signal.find_peaks(corr, height=.5, distance=frame_size)

    used_channels = []
    last_start = None

    for frame_start in frame_starts:
        if last_start is not None:
            # Try salvaging damaged or partial frames
            while last_start + frame_size < frame_start:
                last_start += frame_size
                data[last_start:last_start + frame_size] = rescale_data(
                    data[last_start:last_start + frame_size],
                    low_value,
                    high_value,
                )

        frame_end = frame_start + frame_size
        telemetry = (
            telemetry_data[frame_start:frame_end]
            .reshape((telemetry_values, telemetry_lines))
            .mean(axis=1)
        )

        # Rescale telemetry and image data to be in the intensity range
        high_value = telemetry[7]
        low_value = telemetry[8]

        telemetry = rescale_data(telemetry, low_value, high_value)
        data[frame_start:frame_end] = rescale_data(
            data[frame_start:frame_end],
            low_value,
            high_value,
        )

        # Find which channel is used by comparing the last telemetry value
        # to the initial intensity wedges
        channels = telemetry[:6]
        used_channel = np.abs(channels - telemetry[15]).argmin()
        used_channels.append(used_channel)

        if last_start is None:
            # Try salvaging initial partial frame
            data[:frame_start] = rescale_data(
                data[:frame_start],
                low_value,
                high_value,
            )

        last_start = frame_start

    majority_channel = np.bincount(used_channels).argmax()
    image = Image.fromarray(data[:, sync_width:-telemetry_width].astype("uint8"))
    return majority_channel, image


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
        logging.info(f"Decoding channel {channel}")
        data = data_from_signal(levels, pattern, samples_per_symbol)
        channel_id, image = read_telemetry(data)

        logging.info(f"Channel {channel} is #{channel_id}")
        channels.append(image)

    return channels
