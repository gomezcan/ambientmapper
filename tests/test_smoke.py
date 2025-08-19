def test_import_and_version():
    import ambientmapper
    assert hasattr(ambientmapper, "__version__")
