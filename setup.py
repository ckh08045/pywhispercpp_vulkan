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
        is_vulkan = os.environ.get("WHISPER_VULKAN", "0") == "1"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ] + get_cmake_flags()

        # ── Windows: Visual Studio 제너레이터 명시 ──────────────────────
        # ExternalProject_Add(vulkan-shaders-gen) 가 서브 cmake 인스턴스를
        # 띄울 때 부모 제너레이터를 상속받지 못해 MSVC를 못 찾는 문제 방지.
        # 제너레이터를 명시하면 ExternalProject도 동일 제너레이터로 configure됨.
        # if sys.platform == "win32":
        #     cmake_args += ["-G", "Visual Studio 17 2022", "-A", "x64"]

        # build_args = ["--config", build_type]
        # if sys.platform == "win32":
        #     build_args += ["--", "/m"]
        # else:
        #     # Vulkan 셰이더 병렬 컴파일은 메모리를 많이 소비
        #     # GHA runner 7GB 기준: vulkan=-j2, cpu=-j4
        #     jobs = 2 if is_vulkan else min(os.cpu_count() or 2, 4)
        #     build_args += ["--", f"-j{jobs}"]

        
        # 하위 서브모듈(vulkan-shaders-gen 등) 빌드 시 MSVC 환경 변수 
        # 상속 누락 버그를 방지하기 위해 Ninja 빌드 시스템을 사용합니다.
        if sys.platform == "win32":
            # Ninja를 사용하고, 64비트(x64) 빌드임을 명시
            cmake_args += [
                "-G", "Ninja",
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
            ]

        build_args = ["--config", build_type]
        
        # Windows(Ninja)와 Linux 모두 동일하게 -j 옵션으로 병렬 빌드 수행
        jobs = 2 if is_vulkan else min(os.cpu_count() or 2, 4)
        build_args += ["--", f"-j{jobs}"]
        
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # .resolve()로 절대경로 강제 (Windows에서 '.'으로 전달되는 문제 방지)
        source_dir = Path(__file__).parent.resolve()

        print(f"[setup.py] source_dir : {source_dir}")
        print(f"[setup.py] build_temp : {build_temp}")
        print(f"[setup.py] cmake_args : {cmake_args}")

        subprocess.run(
            ["cmake", str(source_dir)] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", "--verbose"] + build_args,
            cwd=build_temp,
            check=True,
        )
        
def get_version() -> str:
    base = "1.4.1"
    if os.environ.get("WHISPER_VULKAN", "0") == "1":
        return f"{base}+vulkan"
    elif os.environ.get("WHISPER_OPENBLAS", "0") == "1":
        return f"{base}+openblas"
    else:
        return f"{base}+cpu"


def get_package_name() -> str:
    if os.environ.get("WHISPER_VULKAN", "0") == "1":
        return "pywhispercpp-vulkan"
    elif os.environ.get("WHISPER_OPENBLAS", "0") == "1":
        return "pywhispercpp-openblas"
    else:
        return "pywhispercpp-cpu"


setup(
    name=get_package_name(),
    version=get_version(),   # "1.4.1.vulkan" 또는 "1.4.1.cpu
    author="abdeladim-s (fork)",
    description="Python bindings for whisper.cpp (custom build)",
    packages=["pywhispercpp"],
    ext_modules=[Extension("pywhispercpp._pywhispercpp", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=["numpy"],
    zip_safe=False,
)
