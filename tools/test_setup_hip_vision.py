"""Windows HIP vision installation and cache tests. No GPU, compiler or downloads.

    python -m unittest tools.test_setup_hip_vision
"""
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup


class Install(unittest.TestCase):
    def test_install_reuse_and_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            eng = root / "engine"
            eng.mkdir()
            original = {"backend": "hip", "lib_dirs": ["rocm/bin"], "rocm": "engine-runtime"}
            stamp = eng / "BUILD.json"
            stamp.write_text(json.dumps(original))
            sdk_root = root / "sdk"
            compiler = sdk_root / "lib/llvm/bin/clang++.exe"
            compiler.parent.mkdir(parents=True)
            compiler.touch()
            (sdk_root / "bin").mkdir()
            for name in ("amdhip64_7.dll", "amd_comgr.dll", "rocm_kpack.dll"):
                (sdk_root / "bin" / name).write_bytes(name.encode())
            calls = []

            def run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                if "venv" in cmd:
                    scripts = root / ".rocm-win/Scripts"
                    scripts.mkdir(parents=True)
                    (scripts / "python.exe").touch()
                    (scripts / "rocm-sdk.exe").touch()
                if cmd[0] == "cmd.exe":
                    binary = root / "build-vision-hip/bin" / setup.VEXE
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(b"encoder")

            env = {k: v for k, v in os.environ.items()
                   if k not in ("ROCM_VENV", "STRATA_ROCM_VERSION", "STRATA_ROCM_INDEX")}
            with mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "run", run), \
                    mock.patch.object(setup, "out", lambda cmd: str(sdk_root)), \
                    mock.patch.object(setup, "find_vcvars", lambda: root / "vcvars64.bat"), \
                    mock.patch.object(setup, "source_hash", lambda parts: "source-one"), \
                    mock.patch.object(setup, "hip_runtime_closure", lambda p: ["amdhip64_7.dll", "rocm_kpack.dll"]), \
                    mock.patch.dict(os.environ, env, clear=True):
                gpu = {"arch": "gfx1100"}
                setup.build_windows_hip_vision(eng, gpu, root / "llama")
                meta = json.loads(stamp.read_text())
                for k, v in original.items():
                    self.assertEqual(meta[k], v)  # language engine metadata/runtime unchanged
                self.assertEqual(meta["vision_backend"], "hip")
                self.assertEqual(meta["vision_archs"], ["gfx1100"])
                self.assertEqual(meta["vision_lib_dirs"], [str(sdk_root / "bin")])
                self.assertEqual(Path(meta["vision_exe"]).read_bytes(), b"encoder")
                self.assertTrue((eng / "vision-hip/rocm_kpack.dll").is_file())
                self.assertFalse((eng / "amdhip64_7.dll").exists())
                self.assertTrue(any("rocm[libraries,devel,device-gfx1100]==10.2.0a20260930" in c for c, _ in calls))
                calls.clear()
                setup.build_windows_hip_vision(eng, gpu, root / "llama")
                self.assertEqual(calls, [])
                (eng / "vision-hip/rocm_kpack.dll").unlink()
                setup.build_windows_hip_vision(eng, gpu, root / "llama")
                self.assertEqual(calls, [])
                self.assertTrue((eng / "vision-hip/rocm_kpack.dll").is_file())
                # Changing mode, source, architectures, SDK version, or an engine update rebuilds/redeploys.
                for key, value in (("vision_backend", "cpu"), ("vision_src", "old"),
                                   ("vision_archs", ["gfx1201"]), ("vision_rocm", "old"),
                                   ("vision_llama", "old")):
                    changed = json.loads(stamp.read_text())
                    changed[key] = value
                    stamp.write_text(json.dumps(changed))
                    calls.clear()
                    setup.build_windows_hip_vision(eng, gpu, root / "llama")
                    self.assertEqual(len(calls), 1, key)
                    self.assertEqual(calls[0][1]["env"]["STRATA_HIP_ARCHS"], "gfx1100")
                Path(meta["vision_exe"]).unlink()
                stamp.write_text(json.dumps(original))
                calls.clear()
                setup.build_windows_hip_vision(eng, gpu, root / "llama")
                self.assertEqual(len(calls), 1)

    def test_missing_compiler_stops_before_downloads(self):
        with mock.patch.object(setup, "find_vcvars", lambda: None), mock.patch.object(setup, "run") as run:
            with self.assertRaises(SystemExit):
                setup.build_windows_hip_vision(Path("engine"), {"arch": "gfx1100"}, Path("llama"))
            run.assert_not_called()

    def test_setup_downloads_projector_and_writes_encoder_config_only_when_opted_in(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_setup_golden import install
        card = {"index": 0, "name": "AMD Radeon RX 7900 XTX", "vram_gb": 24.0,
                "arch": "gfx1100", "driver": "hip", "count": 1}
        builds = []

        def prebuilt(*args, **kwargs):
            eng = setup.ROOT / "engine"
            (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.42", "source": "prebuilt",
                "backend": "hip", "archs": ["gfx1100"], "lib_dirs": ["rocm/bin"]}))
            return eng

        def vision(eng, gpu, llama):
            builds.append(gpu)
            stamp = eng / "BUILD.json"
            meta = json.loads(stamp.read_text())
            meta.update(vision_exe=str(eng / "vision-hip" / setup.VEXE),
                        vision_lib_dirs=[str(setup.ROOT / ".rocm-win/sdk/bin")])
            stamp.write_text(json.dumps(meta))
            return eng

        extra = [mock.patch.object(setup, "WIN", True),
                 mock.patch.object(setup, "amd_gpus", lambda: [card]),
                 mock.patch.object(setup, "get_prebuilt_hip", prebuilt),
                 mock.patch.object(setup, "hip_card", lambda *a: card),
                 mock.patch.object(setup, "build_windows_hip_vision", vision)]
        args = ["--backend", "hip", "--family", "qwen", "--model", "IQ3_XXS", "--no-start"]
        code, out, cfg, _ = install(63.7, [], [*args, "--vision", "gpu"], extra=extra)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(builds), 1)
        self.assertTrue(cfg["vision"]["gpu"])
        self.assertIn("vision-hip", cfg["vision"]["exe"])
        self.assertEqual(cfg["vision"]["lib_dirs"], ["<T>/.rocm-win/sdk/bin"])
        self.assertIn("mmproj-Qwen3.8-Flash-Next-BF16.gguf", cfg["vision"]["mmproj"])
        self.assertIn("vision encoder:", out)  # emitted after the existing downloader succeeds
        self.assertIn("--vision", cfg["args"])
        code, out, cfg, _ = install(63.7, [], args, extra=extra)
        self.assertEqual(code, 0, out)
        self.assertEqual(len(builds), 1)
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--vision", cfg["args"])


if __name__ == "__main__":
    unittest.main()
