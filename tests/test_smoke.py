def test_import_and_version():
    import ambientmapper
    assert hasattr(ambientmapper, "__version__")

def test_cli_help_runs():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "--version"])  # sanity
    subprocess.check_call(["ambientmapper", "--help"])
