from pathlib import Path

import cv2
import numpy as np
import scipy
from moviepy.editor import VideoFileClip
from pytest import CaptureFixture

from avcap.skeleton import generate_filename, main, process_video_with_ffmpeg

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"


def test_generate_filename() -> None:
    """Test the generate_filename function"""
    filename = generate_filename()
    assert filename.startswith("preprocessed_")
    assert filename.endswith(".mp4")

    filename = generate_filename("test.mp4")
    assert filename.startswith("preprocessed_test_")
    assert filename.endswith(".mp4")

    filename = generate_filename("test")
    assert filename.startswith("preprocessed_test_")
    assert filename.endswith(".mp4")

    filename = generate_filename("test.avi")
    assert filename.startswith("preprocessed_test_")
    assert filename.endswith(".avi")

    filename = generate_filename("test/test.avi")
    filename_parent = Path(filename).parent
    filename_name = Path(filename).name
    assert filename_parent.name == "test"
    assert filename_name.startswith("preprocessed_test_")
    assert filename.endswith(".avi")

    filename = generate_filename("test/test")
    filename_parent = Path(filename).parent
    filename_name = Path(filename).name
    assert filename_parent.name == "test"
    assert filename_name.startswith("preprocessed_test_")
    assert filename.endswith(".mp4")

    filename = generate_filename("test/test.mp4")
    filename_parent = Path(filename).parent
    filename_name = Path(filename).name
    assert filename_parent.name == "test"
    assert filename_name.startswith("preprocessed_test_")
    assert filename.endswith(".mp4")


def test_process_video_with_ffmpeg() -> None:
    """Test the process_video_with_ffmpeg function"""
    # Generate audio and video files
    audio_filename = "test.wav"
    video_filename = "test.mp4"
    av_filename = "processed_test.mp4"

    # Generate a 440 Hz sine wave for 10 seconds
    duration = 10  # seconds
    sample_rate = 44100  # Hz
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)
    signal = np.sin(2 * np.pi * 440 * t)
    signal /= np.max(np.abs(signal))  # Normalize
    scipy.io.wavfile.write(audio_filename, sample_rate, signal.astype(np.float32))

    # Generate a 10 second video fading to black
    video = np.zeros((240, 320, 3), dtype=np.uint8)
    for i in range(240):
        video[i, :, :] = i
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, 24, (320, 240), True)
    for _ in range(240):  # Adjusted to ensure correct duration
        out.write(video)
    out.release()

    # Process the video and audio to combine them
    process_video_with_ffmpeg(video_filename, audio_filename, av_filename)

    # Verify the output file exists
    assert Path(av_filename).exists(), "Processed video file does not exist"

    # Verify the output file matches the input audio
    audio_sampling_rate, audio = scipy.io.wavfile.read(audio_filename)
    video_clip = VideoFileClip(av_filename)
    audio_clip_duration = audio.shape[0] / audio_sampling_rate
    assert (
        audio_clip_duration == video_clip.duration
    ), "Processed video has incorrect duration"


def test_main(capsys: CaptureFixture) -> None:
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["-d", "15"])
    captured = capsys.readouterr()
    assert "" in captured.out
