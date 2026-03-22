import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def get_cmake_flags() -> list:
    flags = [
        "-DGGML_NATIVE=OFF",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_SERVER=OFF",
        "-DWHISPER_SDL2=OFF",
    ]

    if os.environ.get("WHISPER_VULKAN", "0") == "1":
        flags.append("-DGGML_VULKAN=ON")
        print("[setup.py] Backend: Vulkan")
    elif os.environ.get("WHISPER_OPENBLAS", "0") == "1":
        flags.append("-DGGML_BLAS=ON")
        flags.append("-DGGML_BLAS_VENDOR=OpenBLAS")
        print("[setup.py] Backend: OpenBLAS")
    else:
        print("[setup.py] Backend: CPU only")

    extra = os.environ.get("WHISPER_EXTRA_CMAKE_ARGS", "")
    if extra:
        flags.extend(extra.split())

    return flags


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        ext_fullpath = Path(self.get_ext_fullpath(ext.name))
        extdir = ext_fullpath.parent.resolve()

        build_type = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ] + get_cmake_flags()

        build_args = ["--config", build_type]
        if sys.platform != "win32":
            # Linux: nproc 기반, OOM 방지를 위해 최대 4로 제한
            jobs = min(os.cpu_count() or 2, 4)
            build_args += ["--", f"-j{jobs}"]
        else:
            build_args += ["--", "/m"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # ★ 핵심 픽스: .resolve()로 절대경로 강제
        # Windows에서 __file__ 이 상대경로('.') 일 때
        # cmake 가 build_temp 기준으로 경로를 해석해서 CMakeLists.txt 를 못 찾는 문제 방지
        source_dir = Path(__file__).parent.resolve()

        print(f"[setup.py] source_dir  : {source_dir}")
        print(f"[setup.py] build_temp  : {build_temp}")
        print(f"[setup.py] cmake_args  : {cmake_args}")

        # Configure
        subprocess.run(
            ["cmake", str(source_dir)] + cmake_args,
            cwd=build_temp,
            check=True,
        )

        # Build — verbose 출력으로 에러 위치 파악 용이
        subprocess.run(
            ["cmake", "--build", ".", "--verbose"] + build_args,
            cwd=build_temp,
            check=True,
        )


def get_package_name() -> str:
    if os.environ.get("WHISPER_VULKAN", "0") == "1":
        return "pywhispercpp-vulkan"
    elif os.environ.get("WHISPER_OPENBLAS", "0") == "1":
        return "pywhispercpp-openblas"
    else:
        return "pywhispercpp-cpu"


setup(
    name=get_package_name(),
    version="1.2.0",
    author="abdeladim-s (fork)",
    description="Python bindings for whisper.cpp (custom build)",
    packages=["pywhispercpp"],
    ext_modules=[Extension("pywhispercpp._pywhispercpp", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=["numpy"],
    zip_safe=False,
)
