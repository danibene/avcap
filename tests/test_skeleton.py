from avcap.skeleton import main

__author__ = "danibene"
__copyright__ = "danibene"
__license__ = "MIT"


def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["10"])
    captured = capsys.readouterr()
    assert "Video capture ends here" in captured.out