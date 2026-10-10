"""Compare CPU and HIP image embeddings with identical weights and attention settings.

Run with the ROCm SDK's bin directory on PATH. No Python packages are required.
This checks the encoder protocol and reports numerical differences; it does not test image answers.
"""
import argparse
from array import array
import math
from pathlib import Path
import struct
import subprocess


def read_embeddings(path):
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"{path}: missing SVE1 header")
    header = struct.unpack("<5i", data[:20])
    magic, count, nx, ny, width = header
    if magic != 0x31455653 or min(count, nx, ny, width) <= 0 or nx * ny != count:
        raise ValueError(f"{path}: invalid header {header}")
    if len(data) != 20 + count * width * 4:
        raise ValueError(f"{path}: unexpected embedding file size")
    values = array("f")
    values.frombytes(data[20:])
    import sys
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"{path}: non-finite embeddings")
    return header, values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", required=True, type=Path)
    ap.add_argument("--hip", required=True, type=Path)
    ap.add_argument("--mmproj", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("build-vision-hip/comparison"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, exe in (("cpu", a.cpu), ("hip", a.hip)):
        output = (a.out / f"{name}.sve").resolve()
        # The encoder protocol splits the output path at the last space.
        if " " in str(output):
            raise ValueError("Use an output directory without spaces")
        cmd = [str(exe.resolve()), "--mmproj", str(a.mmproj.resolve()), "--model", str(a.model.resolve()),
               "--min-tokens", "1024", "--max-tokens", "1024", "--flash-attn", "off", "--threads", "8"]
        if name == "hip":
            cmd.append("--gpu")
        result = subprocess.run(cmd, input=f"ENC {a.image.resolve()} {output}\nQUIT\n",
                                text=True, capture_output=True, timeout=300)
        (a.out / f"{name}.log").write_text(result.stderr, encoding="utf-8")
        print(f"{name}: {result.stdout.strip()}", flush=True)
        if result.returncode or not result.stdout.startswith("READY ") or "\nOK " not in result.stdout:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {a.out / (name + '.log')}")
        outputs.append(read_embeddings(output))
    (ch, cv), (gh, gv) = outputs
    if ch != gh:
        raise ValueError(f"CPU and HIP geometry differs: {ch} vs {gh}")
    c2 = sum(x * x for x in cv)
    g2 = sum(x * x for x in gv)
    if not c2 or not g2:
        raise ValueError("Encoder returned all-zero embeddings")
    delta = sum((x - y) ** 2 for x, y in zip(cv, gv))
    cosine = sum(x * y for x, y in zip(cv, gv)) / math.sqrt(c2 * g2)
    print(f"tokens={ch[1]}, width={ch[4]}, all values finite")
    print(f"relative RMS error={math.sqrt(delta / c2):.8g}, cosine similarity={cosine:.10f}")
    print(f"maximum absolute difference={max(abs(x-y) for x, y in zip(cv, gv)):.8g}")


if __name__ == "__main__":
    main()
