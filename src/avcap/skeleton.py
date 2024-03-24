import argparse
import logging
import sys
import wave

import cv2
import numpy as np
import sounddevice as sd
from moviepy.editor import AudioFileClip, VideoFileClip

from avcap import __version__

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def capture_video(duration, output_file):
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    if not cap.isOpened():
        _logger.error("Could not open video device")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (width, height), True)

    # Audio recording setup
    channels = 1
    rate = 44100
    audio_output_file = "temp_audio.wav"

    # Prepare to record audio in a separate thread
    audio_frames = []

    def callback(indata, frames, time, status):
        audio_frames.append(indata.copy())

    # Start audio recording
    with sd.InputStream(samplerate=rate, channels=channels, callback=callback):
        start_time = cv2.getTickCount()
        while True:
            ret, frame = cap.read()
            if not ret:
                _logger.error("Failed to capture frame")
                break

            out.write(frame)
            cv2.imshow("frame", frame)

            if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > duration:
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save the recorded audio to a file
    audio_data = np.concatenate(audio_frames, axis=0)
    with wave.open(audio_output_file, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # Assuming 16 bits/sample, change if different
        wf.setframerate(rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    return audio_output_file


def process_video_with_moviepy(input_filepath, output_filepath, audio_input):
    video_clip = VideoFileClip(input_filepath)
    audio_clip = AudioFileClip(audio_input)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(
        output_filepath, fps=24, codec="libx264", audio_codec="aac"
    )


def parse_args(args):
    """Parse command line parameters for video capture."""
    parser = argparse.ArgumentParser(
        description="Capture video from webcam and process it with moviepy"
    )
    parser.add_argument("--version", action="version", version=f"avcap {__version__}")
    parser.add_argument(
        dest="duration",
        help="Duration of the video capture in seconds",
        type=int,
        metavar="SECONDS",
    )
    parser.add_argument(
        dest="output_file",
        help="Path to the output video file",
        type=str,
        metavar="FILE",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging"""
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting video capture...")
    audio_output = capture_video(args.duration, args.output_file)
    _logger.info("Video capture complete. Processing video with moviepy...")
    process_video_with_moviepy(
        args.output_file, "processed_" + args.output_file, audio_output
    )
    _logger.info("Video processing complete")


def run():
    """Entry point for the script."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
