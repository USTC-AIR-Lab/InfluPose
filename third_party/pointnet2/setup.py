"""Build the PointNet++ operators used by the reference checkpoints."""

from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
SOURCES = [
    "pointnet2_api.cpp",
    "ball_query.cpp",
    "ball_query_gpu.cu",
    "group_points.cpp",
    "group_points_gpu.cu",
    "interpolate.cpp",
    "interpolate_gpu.cu",
    "sampling.cpp",
    "sampling_gpu.cu",
]

setup(
    name="influpose-point-ops",
    version="0.1.0",
    py_modules=["pointnet2_utils"],
    ext_modules=[
        CUDAExtension(
            "pointnet2_cuda",
            [str(ROOT / "src" / name) for name in SOURCES],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
