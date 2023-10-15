"""
Decoding algorithms for Automatic Picture Transmission (APT) mode.

Sources:

- <https://noaasis.noaa.gov/NOAASIS/pubs/Users_Guide-Building_Receive_Stations_March_2009.pdf>
- <https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)>
"""
import math
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import scipy.signal
import scipy.io
import scipy.interpolate
import logging
from .xcorr import correlate_template


# Human-readable names for the channels of the AVHRR instrument
channels_names = {
    0: "Channel 1 - Visible",
    1: "Channel 2 - Near-Infrared",
    2: "Channel 3A - Near-infrared",
    3: "Channel 4 - Thermal",
    4: "Channel 5 - Thermal",
    5: "Channel 3B - Thermal",
}

# Spectral range (in μm) of the channels of the AVHRR instrument
channels_spectrums = {
    0: (0.58, 0.68),
    1: (0.725, 1.0),
    2: (1.58, 1.64),
    3: (10.3, 11.3),
    4: (11.5, 12.5),
    5: (3.55, 3.93),
}


@dataclass(frozen=True, slots=True)
class Channel:
    name: str
    spectrum: tuple[float, float]
    image: Image


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


# Number of symbols in a complete APT line
line_width = 1040

# Number of APT lines transmitted every second
lines_per_second = 4

# Carrier frequency used to transmit the signal
carrier_frequency = 2400

# Distance seen on either side of the satellite (in meters)
span = 1_400_000

# Synchronization pattern for start of line in channel A
sync_a_pattern = "000011001100110011001100110011000000000"

# Synchronization pattern for start of line in channel B
sync_b_pattern = "000011100111001110011100111001110011100"

# Total width of the line and time sync parts of a frame
sync_width = len(sync_a_pattern) + 47

# Synchronization pattern for start of frame in both channels
frame_pattern = gen_sync_signal("123456780", samples_per_symbol=8)

# Number of telemetry values in each frame
telemetry_values = 16

# Target intensity values in telemetry
telemetry_targets = np.array([0, 31, 61, 95, 127, 159, 191, 223, 255])

# Number of lines for each telemetry value in each frame
telemetry_lines = 8

# Total length of each frame in lines
frame_size = telemetry_values * telemetry_lines

# Width of a telemetry value in each line
telemetry_width = 45


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
    Decode data from a channel of an APT signal.

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


def rescale_data(data, initial, target):
    """Interpolate data to map initial values onto target values."""
    filter_initial = [initial[0]]
    filter_target = [target[0]]

    # Only keep increasing initial values
    index = 1

    while index < len(initial):
        while index < len(initial) and initial[index] < filter_initial[-1]:
            index += 1

        filter_initial.append(initial[index])
        filter_target.append(target[index])
        index += 1

    interp = scipy.interpolate.CubicSpline(filter_initial, filter_target)
    return interp(data).round().clip(target[0], target[-1])


def read_channel(data):
    """
    Read frame telemetry and image from a decoded APT channel.

    :param data: decoded channel data.
    :returns: structured telemetry and image information.
    """
    data = data.copy()

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
                    initial_values,
                    telemetry_targets,
                )

        frame_end = frame_start + frame_size
        telemetry = (
            telemetry_data[frame_start:frame_end]
            .reshape((telemetry_values, telemetry_lines))
            .mean(axis=1)
        )
        initial_values = np.concatenate(([telemetry[8]], telemetry[:8]))

        # Interpolate image data to be in the 0-255 intensity range
        data[frame_start:frame_end] = rescale_data(
            data[frame_start:frame_end],
            initial_values,
            telemetry_targets,
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
                initial_values,
                telemetry_targets,
            )

        last_start = frame_start

    majority_channel = np.bincount(used_channels).argmax()
    name = channels_names[majority_channel]
    spectrum = channels_spectrums[majority_channel]
    image = Image.fromarray(data[:, sync_width:-telemetry_width].astype("uint8"))

    return Channel(f"{name} ({spectrum[0]} - {spectrum[1]} μm)", spectrum, image)


def apt_decode(rate, signal):
    """
    Decode both image channels from an APT signal.

    :param rate: signal sampling rate.
    :param signal: list of samples.
    :return: information on the two decoded channels.
    """
    samples_per_symbol = rate / (line_width * lines_per_second)

    logging.info("Demodulating signal")
    levels = amplitude_demod(rate, signal)

    channels = []

    for channel_id, pattern in (("A", sync_a_pattern), ("B", sync_b_pattern)):
        logging.info(f"Decoding channel {channel_id}")
        data = data_from_signal(levels, pattern, samples_per_symbol)
        channel = read_channel(data)
        channels.append(channel)
        logging.info(f"Channel {channel_id} is {channel.name}")

    return channels
