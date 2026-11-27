#!/usr/bin/env python3
"""Download the LiteRT C/C++ SDK headers and prebuilt runtime libraries.

Run once after cloning (and again whenever LITERT_SNAPSHOT below changes):

    py -I tools/fetch_litert_sdk.py                 # headers + host platform
    py -I tools/fetch_litert_sdk.py --platform android_arm64
    py -I tools/fetch_litert_sdk.py --all           # every supported platform

The upstream archive is ~360 MB, but this script reads only the zip's central
directory plus the members it needs (HTTP range requests), so a typical run
downloads 40-60 MB.

Layout produced (both directories are git-ignored):

    litert_cc_sdk/              C/C++ headers + CMake files from the archive
                                (reference for lib/core/litert/ffi.dart)
    litert_runtime/<platform>/  libLiteRt + GPU accelerator libraries that the
                                platform builds bundle next to the app binary
                                (windows/CMakeLists.txt, macos/Runner Xcode
                                build phase, android/app/build.gradle.kts)

On Windows the GPU accelerator (WebGPU/D3D12 via Dawn) also needs the DirectX
Shader Compiler runtime (dxcompiler.dll, dxil.dll). They are copied from the
DirectXShaderCompiler GitHub release (pinned). The Windows SDK copy is only an
offline fallback: older SDK builds are too old for the accelerator.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Dated snapshot of the nightly prebuilts. Headers and runtime libraries must
# come from the same snapshot, so bump this deliberately, never use "latest".
LITERT_SNAPSHOT = "20261126"
LITERT_URL = (
    "https://storage.googleapis.com/litert/binaries/nightly/archive/"
    f"litert_prebuilts_{LITERT_SNAPSHOT}.zip"
)
# DirectX Shader Compiler runtime for the WebGPU/D3D12 accelerator. Dawn needs a
# recent build: the dxcompiler.dll in older Windows SDKs fails device creation
# with E_NOINTERFACE, so pin a release instead of using the SDK copy.
DXC_URL = (
    "https://github.com/microsoft/DirectXShaderCompiler/releases/download/"
    "v1.9.2609/dxc_2026_09_29.zip"
)

ROOT = Path(__file__).resolve().parent.parent
SDK_DIR = ROOT / "litert_cc_sdk"
RUNTIME_DIR = ROOT / "litert_runtime"

# Archive sub-directories we know how to bundle.
PLATFORMS = ("windows_x86_64", "macos_arm64", "android_arm64")
RUNTIME_SUFFIXES = (".dll", ".dylib", ".so")
# Vendor NPU plugins are not used by this app; skip them to save space.
SKIP_MARKERS = ("Dispatch_", "CompilerPlugin_")
# LiteRT registers the first GPU accelerator library that loads, so Android
# bundles only the OpenCL/GL one (the WebGPU one is for Windows/Linux).
SKIP_BY_PLATFORM = {"android_arm64": ("WebGpu",)}
# Gradle's jniLibs wants <dir>/<abi>/lib*.so.
SUBDIR_BY_PLATFORM = {"android_arm64": "arm64-v8a"}


def host_platform() -> str:
    if sys.platform == "win32":
        return "windows_x86_64"
    if sys.platform == "darwin":
        return "macos_arm64"
    raise SystemExit("Unsupported host; pass --platform explicitly.")


class RangeFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        out = bytearray()
        # Large members are fetched in chunks so a dropped connection only
        # costs one chunk.
        while n > 0:
            take = min(n, 8 << 20)
            end = self.pos + take - 1
            req = urllib.request.Request(
                self.url, headers={"Range": f"bytes={self.pos}-{end}"}
            )
            with urllib.request.urlopen(req) as r:
                chunk = r.read()
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def extract(z: zipfile.ZipFile, info: zipfile.ZipInfo, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with z.open(info) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)


def runtime_members(z: zipfile.ZipFile, platform: str) -> list[zipfile.ZipInfo]:
    prefix = platform + "/"
    found = []
    for info in z.infolist():
        name = info.filename
        if not name.startswith(prefix) or info.is_dir():
            continue
        base = name[len(prefix):]
        if "/" in base or not base.endswith(RUNTIME_SUFFIXES):
            continue
        if any(m in base for m in SKIP_MARKERS + SKIP_BY_PLATFORM.get(platform, ())):
            continue
        found.append(info)
    return found


def fetch_sdk(force: bool, platforms: list[str]) -> None:
    stamp = RUNTIME_DIR / ".snapshot"
    wanted = set(platforms)
    have = set()
    if stamp.exists() and not force:
        data = json.loads(stamp.read_text())
        if data.get("snapshot") == LITERT_SNAPSHOT and SDK_DIR.exists():
            have = set(data.get("platforms", []))
    missing = wanted - have
    if not missing and SDK_DIR.exists():
        print(f"LiteRT {LITERT_SNAPSHOT} already present for {sorted(wanted)}.")
        return

    print(f"Reading {LITERT_URL}")
    z = zipfile.ZipFile(RangeFile(LITERT_URL))

    if force or not SDK_DIR.exists() or not (have):
        if SDK_DIR.exists():
            shutil.rmtree(SDK_DIR)
        n = 0
        for info in z.infolist():
            if info.filename.startswith("litert_cc_sdk/") and not info.is_dir():
                extract(z, info, ROOT / info.filename)
                n += 1
        print(f"  headers: {n} files -> {SDK_DIR.relative_to(ROOT)}/")

    for platform in sorted(missing):
        root = RUNTIME_DIR / platform
        dest = root / SUBDIR_BY_PLATFORM.get(platform, "")
        if root.exists():
            shutil.rmtree(root)
        members = runtime_members(z, platform)
        if not members:
            raise SystemExit(f"No runtime libraries for {platform} in archive.")
        for info in members:
            print(f"  {info.filename} ({info.file_size / 1e6:.1f} MB)")
            extract(z, info, dest / Path(info.filename).name)
        if platform == "windows_x86_64":
            fetch_dxc(dest)

    RUNTIME_DIR.mkdir(exist_ok=True)
    stamp.write_text(
        json.dumps({"snapshot": LITERT_SNAPSHOT, "platforms": sorted(have | wanted)})
    )


def fetch_dxc(dest: Path) -> None:
    names = ("dxcompiler.dll", "dxil.dll")
    print("  DirectXShaderCompiler runtime")
    try:
        # GitHub's release redirect rejects HEAD, so fetch the 32 MB zip whole.
        with urllib.request.urlopen(DXC_URL, timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        for n in names:
            extract(z, z.getinfo(f"bin/x64/{n}"), dest / n)
        print(f"  {', '.join(names)} from {DXC_URL.rsplit('/', 1)[1]}")
        return
    except Exception as e:  # offline / GitHub unreachable: try the Windows SDK
        print(f"  download failed ({e}); trying the Windows SDK copy")
    kits = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    redist = kits / "Windows Kits" / "10" / "Redist" / "D3D" / "x64"
    if not all((redist / n).exists() for n in names):
        raise SystemExit("dxcompiler.dll / dxil.dll unavailable (no network, no Windows SDK).")
    for n in names:
        shutil.copy2(redist / n, dest / n)
    print(f"  copied from {redist} (may be too old for the GPU accelerator)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--platform", action="append", choices=PLATFORMS,
        help="runtime platform(s) to fetch (default: host)",
    )
    ap.add_argument("--all", action="store_true", help="fetch every platform")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args()
    platforms = list(PLATFORMS) if args.all else (args.platform or [host_platform()])
    fetch_sdk(args.force, platforms)
    print("Done.")


if __name__ == "__main__":
    main()
