import argparse
import logging
import sys

import cv2
from moviepy.editor import VideoFileClip

from avcap import __version__

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


def capture_video(duration, output_file):
    """
    Captures video from the default webcam and saves it to a file.

    Args:
      duration (int): Duration of the video capture in seconds.
      output_file (str): Path to the output file where the video will be saved.
    """
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    if not cap.isOpened():
        _logger.error("Could not open video device")
        return

    # Determine the video width and height by querying the capture device
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Attempt using H264 codec; you might need to adjust this based on your system's support
    # For Windows, you might use 'H264' or 'XVID' and for macOS, 'avc1' or 'mp4v' could work
    fourcc = cv2.VideoWriter_fourcc(
        *"avc1"
    )  # Try using 'avc1' or 'h264' instead of 'XVID'
    out = cv2.VideoWriter(
        output_file, fourcc, 20.0, (width, height), True
    )  # Ensure isColor=True for color videos

    start_time = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        if not ret:
            _logger.error("Failed to capture frame")
            return

        out.write(frame)  # Save the captured frame
        cv2.imshow("frame", frame)  # Display the frame

        # Stop recording after 'duration' seconds
        if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > duration:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Allow quitting with the 'q' key
            break

    cap.release()  # Release the webcam
    out.release()  # Close the file being written to
    cv2.destroyAllWindows()  # Close the window showing the frame


def process_video_with_moviepy(input_filepath, output_filepath):
    """
    Processes a video file using moviepy to rewrite it with specific encoding settings.

    Args:
      input_filepath (str): Path to the input video file.
      output_filepath (str): Path to the output video file.
    """
    video_clip = VideoFileClip(input_filepath)
    video_clip.write_videofile(
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
    """Main function adjusted for capturing and processing video."""
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting video capture...")
    capture_video(args.duration, args.output_file)
    _logger.info("Video capture complete. Processing video with moviepy...")
    process_video_with_moviepy(args.output_file, "processed_" + args.output_file)
    _logger.info("Video processing complete")


def run():
    """Entry point for the script."""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
