from pytest import CaptureFixture

from avcap.skeleton import generate_filename, main

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
    assert filename.startswith("test/preprocessed_test_")
    assert filename.endswith(".avi")

    filename = generate_filename("test/test")
    assert filename.startswith("test/preprocessed_test_")
    assert filename.endswith(".mp4")

    filename = generate_filename("test/test.mp4")
    assert filename.startswith("test/preprocessed_test_")
    assert filename.endswith(".mp4")


def test_main(capsys: CaptureFixture) -> None:
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["-d", "15"])
    captured = capsys.readouterr()
    assert "Video capture ends here" in captured.out
