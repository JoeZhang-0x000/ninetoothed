class NineToothedError(Exception):
    pass


class NineToothedLaunchError(NineToothedError):
    def __init__(
        self,
        message,
        *,
        kernel_name=None,
        source_path=None,
        grid=None,
        num_warps=None,
        num_stages=None,
        required_programs=None,
        max_grid_size=None,
    ):
        super().__init__(message)
        self.kernel_name = kernel_name
        self.source_path = source_path
        self.grid = grid
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.required_programs = required_programs
        self.max_grid_size = max_grid_size


class NineToothedGridError(NineToothedLaunchError):
    pass
