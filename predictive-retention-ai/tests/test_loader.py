from src.data.loader import DATASET_FILENAME, resolve_raw_data_path


def test_default_dataset_path_can_be_resolved():
    path = resolve_raw_data_path()

    assert path.exists()
    assert path.name == DATASET_FILENAME


def test_explicit_missing_dataset_has_clear_error(tmp_path):
    missing = tmp_path / DATASET_FILENAME

    try:
        resolve_raw_data_path(missing)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Missing dataset should raise FileNotFoundError")
