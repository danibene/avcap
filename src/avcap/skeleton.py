import argparse
import logging
import sys
import cv2

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
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))
    
    start_time = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        if not ret:
            _logger.error("Failed to capture frame")
            break
        
        out.write(frame)
        cv2.imshow('frame', frame)
        
        # Stop recording after 'duration' seconds
        if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > duration:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Other parts of the skeleton remain unchanged

def parse_args(args):
    """Parse command line parameters for video capture

    This is adjusted for video capturing parameters.
    """
    parser = argparse.ArgumentParser(description="Capture video from webcam")
    parser.add_argument("--version", action="version", version=f"avcap {__version__}")
    parser.add_argument(dest="duration", help="Duration of the video capture in seconds", type=int, metavar="SECONDS")
    parser.add_argument(dest="output_file", help="Path to the output video file", type=str, metavar="FILE")
    parser.add_argument("-v", "--verbose", dest="loglevel", help="set loglevel to INFO", action="store_const", const=logging.INFO)
    parser.add_argument("-vv", "--very-verbose", dest="loglevel", help="set loglevel to DEBUG", action="store_const", const=logging.DEBUG)
    return parser.parse_args(args)

def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Main function adjusted for capturing video from the webcam."""
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting video capture...")
    capture_video(args.duration, args.output_file)
    _logger.info("Video capture ends here")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html
    run()
