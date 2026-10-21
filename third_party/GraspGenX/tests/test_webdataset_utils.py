import json
import os
import pytest
from graspgenx.dataset.webdataset_utils import is_webdataset, load_uuid_list


class TestIsWebdataset:
    def test_empty_dir_is_not_webdataset(self, tmp_path):
        assert is_webdataset(str(tmp_path)) is False

    def test_nonexistent_dir_is_not_webdataset(self):
        assert is_webdataset("/nonexistent/path") is False

    def test_dir_with_tar_is_webdataset(self, tmp_path):
        (tmp_path / "shard_000.tar").write_bytes(b"")
        assert is_webdataset(str(tmp_path)) is True

    def test_dir_with_json_only_is_not_webdataset(self, tmp_path):
        (tmp_path / "data.json").write_text("{}")
        assert is_webdataset(str(tmp_path)) is False

    def test_dir_with_mixed_files(self, tmp_path):
        (tmp_path / "shard_000.tar").write_bytes(b"")
        (tmp_path / "uuid_index.json").write_text("{}")
        assert is_webdataset(str(tmp_path)) is True


class TestLoadUuidList:
    def test_load_from_json_list(self, tmp_path):
        uuids = ["uuid-1", "uuid-2", "uuid-3"]
        json_file = tmp_path / "uuids.json"
        json_file.write_text(json.dumps(uuids))

        result = load_uuid_list(str(json_file))
        assert result == uuids

    def test_load_from_json_dict(self, tmp_path):
        uuid_dict = {"uuid-1": 0, "uuid-2": 1}
        json_file = tmp_path / "uuids.json"
        json_file.write_text(json.dumps(uuid_dict))

        result = load_uuid_list(str(json_file))
        assert set(result) == {"uuid-1", "uuid-2"}

    def test_load_from_txt(self, tmp_path):
        txt_file = tmp_path / "uuids.txt"
        txt_file.write_text("uuid-1\nuuid-2\nuuid-3\n")

        result = load_uuid_list(str(txt_file))
        assert result == ["uuid-1", "uuid-2", "uuid-3"]

    def test_load_from_txt_strips_whitespace(self, tmp_path):
        txt_file = tmp_path / "uuids.txt"
        txt_file.write_text("  uuid-1  \n  uuid-2  \n")

        result = load_uuid_list(str(txt_file))
        assert result == ["uuid-1", "uuid-2"]

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_uuid_list("/nonexistent/uuids.json")

    def test_load_unsupported_format_raises(self, tmp_path):
        csv_file = tmp_path / "uuids.csv"
        csv_file.write_text("uuid-1,uuid-2\n")

        with pytest.raises(ValueError, match="Unsupported"):
            load_uuid_list(str(csv_file))
