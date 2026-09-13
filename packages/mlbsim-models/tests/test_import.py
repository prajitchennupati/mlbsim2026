import mlbsim_models


def test_package_version_exposed():
    assert mlbsim_models.__version__ == "0.0.0"
