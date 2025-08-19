import os, subprocess, sys, tempfile, json, pathlib

def test_cli_help_runs():
    subprocess.check_call(["ambientmapper", "run", "--help"])

def test_inline_args_parse(tmp_path: pathlib.Path):
    # create two empty BAMs so existence checks pass
    (tmp_path/"a.bam").write_bytes(b"")
    (tmp_path/"b.bam").write_bytes(b"")
    out = tmp_path/"out"
    cmd = [
        "ambientmapper", "run",
        "--sample", "S",
        "--genome", "G1,G2",
        "--bam", f"{tmp_path/'a.bam'},{tmp_path/'b.bam'}",
        "--workdir", str(out),
        "--threads", "1",
        "--min-barcode-freq", "10",
        "--chunk-size-cells", "1000",
    ]
    # we expect a nonzero exit later in the pipeline (empty BAMs),
    # but the parser & early mode selection should work.
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        # accept failure as long as it got past argument parsing
        assert e.returncode != 2  # Typer uses exit 2 for bad CLI args
