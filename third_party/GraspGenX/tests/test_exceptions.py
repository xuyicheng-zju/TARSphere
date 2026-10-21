from graspgenx.dataset.exceptions import DataLoaderError, ErrorInfo


def test_error_info_dataclass():
    info = ErrorInfo(code=42, description="test error")
    assert info.code == 42
    assert info.description == "test error"


def test_success_error_code():
    assert DataLoaderError.SUCCESS.code == 0
    assert "success" in DataLoaderError.SUCCESS.description.lower()


def test_data_loading_error_codes():
    assert DataLoaderError.GRASPS_FILE_NOT_FOUND.code == 100
    assert DataLoaderError.GRASPS_FILE_LOAD_ERROR.code == 101
    assert DataLoaderError.OBJECT_FILE_NOT_FOUND.code == 106


def test_rendering_error_codes():
    assert DataLoaderError.RENDERING_SUCCESS.code == 200
    assert DataLoaderError.RENDERING_ERROR.code == 201
    assert DataLoaderError.RENDERING_ERROR_POINT_CLOUD_TOO_SMALL.code == 202


def test_error_descriptions_are_nonempty():
    for error in DataLoaderError:
        assert len(error.description) > 0


def test_error_codes_are_integers():
    for error in DataLoaderError:
        assert isinstance(error.code, int)


def test_insufficient_grasps_errors():
    gen_err = DataLoaderError.INSUFFICIENT_GRASPS_FOR_GENERATOR_DATASET
    disc_err = DataLoaderError.INSUFFICIENT_GRASPS_FOR_DISCRIMINATOR_DATASET
    assert gen_err.code == 102
    assert disc_err.code == 103


def test_uuid_errors():
    assert DataLoaderError.UUID_MAPPING_NOT_FOUND.code == 111
    assert DataLoaderError.UUID_NOT_FOUND_IN_MAPPING.code == 112
    assert DataLoaderError.UUID_MAPPING_LOAD_ERROR.code == 113
