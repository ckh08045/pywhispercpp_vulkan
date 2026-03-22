import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# ── 환경변수로 백엔드 선택 ─────────────────────────────────────────
# 사용법:
#   WHISPER_VULKAN=1 pip install .
#   WHISPER_OPENBLAS=1 pip install .
#   (아무것도 없으면 CPU only)
# ─────────────────────────────────────────────────────────────────

def get_cmake_flags() -> list[str]:
    flags = [
        # 공통: 이식성 확보 (빌드 머신 CPU 특화 명령어 비활성화)
        "-DGGML_NATIVE=OFF",
        # 불필요한 빌드 타겟 제거
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

    # 추가 사용자 정의 플래그 (WHISPER_EXTRA_CMAKE_ARGS 환경변수)
    extra = os.environ.get("WHISPER_EXTRA_CMAKE_ARGS", "")
    if extra:
        flags.extend(extra.split())

    return flags


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import cmake

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
            build_args += ["--", f"-j{os.cpu_count() or 2}"]
        else:
            build_args += ["--", "/m"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        source_dir = Path(__file__).parent

        subprocess.run(
            ["cmake", str(source_dir)] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True,
        )


# ── wheel 이름에 백엔드 반영 ─────────────────────────────────────
# whisper_cpp_vulkan-x.x.x-cp311-...whl  또는
# whisper_cpp_cpu-x.x.x-cp311-...whl
def get_package_name() -> str:
    if os.environ.get("WHISPER_VULKAN", "0") == "1":
        return "pywhispercpp-vulkan"
    elif os.environ.get("WHISPER_OPENBLAS", "0") == "1":
        return "pywhispercpp-openblas"
    else:
        return "pywhispercpp-cpu"


setup(
    name=get_package_name(),
    version="1.2.0",           # whisper.cpp 버전과 맞춰서 관리
    author="abdeladim-s (fork)",
    description="Python bindings for whisper.cpp (custom build)",
    packages=["pywhispercpp"],
    ext_modules=[Extension("pywhispercpp._pywhispercpp", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=["numpy"],
    zip_safe=False,
)
