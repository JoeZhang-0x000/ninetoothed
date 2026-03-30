import pytest

from ninetoothed.errors import NineToothedGridError, NineToothedLaunchError
from ninetoothed.runtime import make_launch_grid, wrap_kernel_launch_error


def test_make_launch_grid_keeps_small_launches_1d():
    assert make_launch_grid(17, max_grid_size=(64, 64, 64)) == (17,)


def test_make_launch_grid_spreads_large_launches_across_axes():
    assert make_launch_grid(65, max_grid_size=(64, 64, 64)) == (64, 2)
    assert make_launch_grid(64 * 64 + 1, max_grid_size=(64, 64, 64)) == (64, 64, 2)


def test_make_launch_grid_raises_when_3d_grid_is_insufficient():
    with pytest.raises(NineToothedGridError) as exc_info:
        make_launch_grid(64 * 64 * 64 + 1, max_grid_size=(64, 64, 64))

    error = exc_info.value

    assert error.required_programs == 64 * 64 * 64 + 1
    assert error.max_grid_size == (64, 64, 64)


def test_wrap_kernel_launch_error_wraps_grid_failures():
    class OutOfResources(RuntimeError):
        pass

    error = wrap_kernel_launch_error(
        OutOfResources("out of resource: grid size"),
        kernel_name="attention",
        source_path="/tmp/kernel.py",
        grid=(65535, 65535, 2),
        num_warps=4,
        num_stages=4,
    )

    assert isinstance(error, NineToothedGridError)
    assert error.kernel_name == "attention"
    assert error.grid == (65535, 65535, 2)
    assert "kernel=attention" in str(error)


def test_wrap_kernel_launch_error_wraps_other_launch_failures():
    error = wrap_kernel_launch_error(
        RuntimeError("launch failed"),
        kernel_name="attention",
        source_path="/tmp/kernel.py",
        num_warps=4,
        num_stages=4,
    )

    assert isinstance(error, NineToothedLaunchError)
    assert not isinstance(error, NineToothedGridError)
    assert error.kernel_name == "attention"
    assert "RuntimeError: launch failed" in str(error)
