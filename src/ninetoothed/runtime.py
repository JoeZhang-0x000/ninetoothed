import math

from ninetoothed.errors import NineToothedGridError, NineToothedLaunchError


def get_max_grid_size(max_grid_size=None):
    return _normalize_max_grid_size(max_grid_size)


def get_effective_grid_x_limit(num_warps=1, max_grid_size=None):
    grid_x_limit = get_max_grid_size(max_grid_size)[0]
    num_warps = max(1, int(num_warps))

    return max(1, grid_x_limit // num_warps)


def make_launch_grid(total_programs, max_grid_size=None):
    total_programs = int(total_programs)

    if total_programs < 0:
        raise ValueError("`total_programs` must be non-negative.")

    if total_programs == 0:
        return (0,)

    x_limit, y_limit, z_limit = get_max_grid_size(max_grid_size)

    grid_x = min(total_programs, x_limit)
    remaining_programs = _ceil_div(total_programs, grid_x)

    if remaining_programs == 1:
        return (grid_x,)

    grid_y = min(remaining_programs, y_limit)
    remaining_programs = _ceil_div(remaining_programs, grid_y)

    if remaining_programs == 1:
        return (grid_x, grid_y)

    grid_z = remaining_programs

    if grid_z <= z_limit:
        return (grid_x, grid_y, grid_z)

    raise NineToothedGridError(
        "Kernel launch exceeds the device grid limits.",
        required_programs=total_programs,
        max_grid_size=(x_limit, y_limit, z_limit),
        grid=(grid_x, grid_y, grid_z),
    )


def wrap_kernel_launch_error(
    error,
    *,
    kernel_name,
    source_path,
    grid=None,
    num_warps=None,
    num_stages=None,
):
    if isinstance(error, NineToothedLaunchError):
        message = str(error)
        required_programs = error.required_programs
        max_grid_size = error.max_grid_size
        grid = error.grid if error.grid is not None else grid

        return type(error)(
            _format_launch_error_message(
                message,
                kernel_name=kernel_name,
                source_path=source_path,
                grid=grid,
                num_warps=num_warps,
                num_stages=num_stages,
                required_programs=required_programs,
                max_grid_size=max_grid_size,
            ),
            kernel_name=kernel_name,
            source_path=source_path,
            grid=grid,
            num_warps=num_warps,
            num_stages=num_stages,
            required_programs=required_programs,
            max_grid_size=max_grid_size,
        )

    error_name = type(error).__name__
    error_message = str(error)

    if _is_grid_error(error):
        return NineToothedGridError(
            _format_launch_error_message(
                error_message,
                kernel_name=kernel_name,
                source_path=source_path,
                grid=grid,
                num_warps=num_warps,
                num_stages=num_stages,
            ),
            kernel_name=kernel_name,
            source_path=source_path,
            grid=grid,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return NineToothedLaunchError(
        _format_launch_error_message(
            f"{error_name}: {error_message}",
            kernel_name=kernel_name,
            source_path=source_path,
            grid=grid,
            num_warps=num_warps,
            num_stages=num_stages,
        ),
        kernel_name=kernel_name,
        source_path=source_path,
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _normalize_max_grid_size(max_grid_size):
    if max_grid_size is None:
        max_grid_size = _get_current_max_grid_size()

    if len(max_grid_size) != 3:
        raise ValueError("`max_grid_size` must contain three axes.")

    normalized = tuple(int(limit) for limit in max_grid_size)

    if any(limit <= 0 for limit in normalized):
        raise ValueError("`max_grid_size` must contain positive integers.")

    return normalized


def _get_current_max_grid_size():
    import triton

    device = triton.runtime.driver.active.get_current_device()
    properties = triton.runtime.driver.active.utils.get_device_properties(device)

    if "max_grid_size" in properties:
        return properties["max_grid_size"]

    if "max_grid_sizes" in properties:
        return properties["max_grid_sizes"]

    return (65535, 65535, 65535)


def _ceil_div(lhs, rhs):
    return math.ceil(lhs / rhs)


def _format_launch_error_message(
    message,
    *,
    kernel_name,
    source_path,
    grid,
    num_warps,
    num_stages,
    required_programs=None,
    max_grid_size=None,
):
    details = [f"kernel={kernel_name}", f"source={source_path}"]

    if grid is not None and not callable(grid):
        details.append(f"grid={grid}")

    if num_warps is not None:
        details.append(f"num_warps={num_warps}")

    if num_stages is not None:
        details.append(f"num_stages={num_stages}")

    if required_programs is not None:
        details.append(f"required_programs={required_programs}")

    if max_grid_size is not None:
        details.append(f"max_grid_size={max_grid_size}")

    return f"{message} ({', '.join(details)})"


def _is_grid_error(error):
    error_name = type(error).__name__
    error_message = str(error)

    if error_name == "OutOfResources" and "grid size" in error_message:
        return True

    if isinstance(error, NineToothedGridError):
        return True

    return False
