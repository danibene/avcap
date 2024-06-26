import argparse
import logging
import os
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import ffmpeg
import numpy as np
import sounddevice as sd

from avcap import __version__

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

FPS = 24
DEFAULT_EXTENSION = ".mp4"
PREFIX = "preprocessed_"


def generate_filename(base_name: Optional[str] = None) -> str:
    """Generate a filename based on the current date and time."""
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if not base_name:
        filename = PREFIX + datetime_str + DEFAULT_EXTENSION
    else:
        if not Path(base_name).suffix:
            extension = DEFAULT_EXTENSION
        else:
            extension = Path(base_name).suffix
        parent_path_str = str(Path(base_name).parent)
        if parent_path_str == ".":
            filename = PREFIX + Path(base_name).stem + "_" + datetime_str + extension
        else:
            filename = str(
                Path(
                    parent_path_str,
                    PREFIX + Path(base_name).stem + "_" + datetime_str + extension,
                )
            )
    return filename


def capture_video(
    duration: int, output_file: str, camera_id: int = 0
) -> Tuple[str, str]:
    cap = cv2.VideoCapture(camera_id)  # 0 is usually the default webcam
    if not cap.isOpened():
        _logger.error("Could not open video device")
        return "", ""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_filename = generate_filename(output_file)
    out = cv2.VideoWriter(video_filename, fourcc, float(FPS), (width, height), True)

    # Audio recording setup
    channels = 1
    # Automatically set the sample rate to the one your device supports
    rate = sd.query_devices(None, "input")["default_samplerate"]
    audio_filename = video_filename.replace(Path(video_filename).suffix, ".wav")
    if os.path.exists(audio_filename):
        _logger.error(f"{audio_filename} already exists. Exiting to avoid data loss.")
        return "", ""

    # Prepare to record audio in a separate thread
    audio_frames = []

    def callback(
        indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags
    ) -> None:
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
    with wave.open(audio_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # Assuming 16 bits/sample, change if different
        wf.setframerate(rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    return video_filename, audio_filename


def generate_final_filename(input_filepath: str) -> str:
    processed_filename = Path(input_filepath).name[len(PREFIX) :]
    processed_filepath = str(Path(input_filepath).parent / processed_filename)
    return processed_filepath


def process_video_with_ffmpeg(
    video_filename: str, audio_filename: str, output_filename: str
) -> None:
    """Combine video and audio using ffmpeg-python, saving to output_filename."""
    (
        ffmpeg.input(video_filename)
        .output(
            ffmpeg.input(audio_filename),
            output_filename,
            vcodec="copy",
            acodec="aac",
            strict="experimental",
        )
        .run(overwrite_output=True)
    )


def parse_args(args: list) -> argparse.Namespace:
    """Parse command line parameters for video capture."""
    parser = argparse.ArgumentParser(
        description="Capture video from webcam and process it with moviepy"
    )
    parser.add_argument("--version", action="version", version=f"avcap {__version__}")
    parser.add_argument(
        "-d",
        "--duration",
        dest="duration",
        help="Duration of the video capture in seconds",
        type=int,
        metavar="SECONDS",
        default=10,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Path to the output video file",
        type=str,
        metavar="FILE",
        default="",
    )
    parser.add_argument(
        "-c",
        "--camera",
        dest="camera_id",
        help="Camera ID",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-k",
        "--keep-originals",
        dest="keep_originals",
        help="Whether to keep original audio and video",
        type=bool,
        default=True,
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


def setup_logging(loglevel: int) -> None:
    """Setup basic logging"""
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args: list) -> None:
    parsed_args = parse_args(args)
    setup_logging(parsed_args.loglevel)
    _logger.debug("Starting video capture...")
    video_output, audio_output = capture_video(
        parsed_args.duration, parsed_args.output_file, parsed_args.camera_id
    )
    if not video_output or not audio_output:
        _logger.error("Video capture failed. Exiting...")
        return
    _logger.info("Video capture complete. Processing video with moviepy...")
    processed_filepath = generate_final_filename(video_output)
    process_video_with_ffmpeg(video_output, audio_output, processed_filepath)
    _logger.info("Video processing complete. Cleaning up temporary files...")
    if not parsed_args.keep_originals:
        # Cleanup temporary files
        if os.path.exists(video_output):
            os.remove(video_output)
        if os.path.exists(audio_output):
            os.remove(audio_output)
    _logger.info(f"Final video is available at {processed_filepath}")


def run() -> None:
    """Entry point for the script."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
