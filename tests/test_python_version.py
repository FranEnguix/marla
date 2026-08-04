import pytest

from marla.utils.python_version import UnsupportedPythonVersionError, check_python_version


def test_current_interpreter_is_supported():
    check_python_version()  # should not raise under the pinned 3.10 venv


@pytest.mark.parametrize(
    "version_info",
    [
        (3, 9, 0, "final", 0),
        (3, 11, 0, "final", 0),
        (3, 12, 0, "final", 0),
        (2, 7, 18, "final", 0),
    ],
)
def test_rejects_unsupported_versions(version_info):
    with pytest.raises(UnsupportedPythonVersionError):
        check_python_version(version_info)


def test_accepts_any_310_patch_version():
    check_python_version((3, 10, 99, "final", 0))
