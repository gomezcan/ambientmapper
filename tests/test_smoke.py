import subprocess, sys, pathlib

def test_cli_help_runs():
    subprocess.check_call(["ambientmapper", "run", "--help"])

def test_inline_mode_parsing(tmp_path: pathlib.Path):
    # create two dummy files so existence checks pass; pysam will fail later (that’s ok)
    a = tmp_path / "a.bam"; a.write_bytes(b"")
    b = tmp_path / "b.bam"; b.write_bytes(b"")
    out = tmp_path / "out"

    cmd = [
        "ambientmapper", "run",
        "--sample", "S",
        "--genome", "G1,G2",
        "--bam", f"{a},{b}",
        "--workdir", str(out),
        "--threads", "1",
        "--min-barcode-freq", "10",
        "--chunk-size-cells", "1000",
    ]
    try:
        subprocess.check_call(cmd)
        # If it *does* pass end-to-end in some environments, that’s fine.
    except subprocess.CalledProcessError as e:
        # Typer uses exit code 2 for argument/option parse errors.
        # Any other nonzero return here still proves the CLI mode was parsed.
        assert e.returncode != 2
