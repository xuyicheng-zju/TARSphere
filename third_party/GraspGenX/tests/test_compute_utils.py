import pytest
from graspgenx.utils.compute_utils import _fmt


class TestFmt:
    def test_bytes(self):
        assert "B" in _fmt(500)

    def test_kibibytes(self):
        result = _fmt(2048)
        assert "KiB" in result

    def test_mebibytes(self):
        result = _fmt(5 * 1024 * 1024)
        assert "MiB" in result

    def test_gibibytes(self):
        result = _fmt(3 * 1024 ** 3)
        assert "GiB" in result

    def test_tebibytes(self):
        result = _fmt(2 * 1024 ** 4)
        assert "TiB" in result

    def test_zero(self):
        result = _fmt(0)
        assert "0.00 B" == result

    def test_negative(self):
        result = _fmt(-1024)
        assert "KiB" in result

    def test_exact_boundary(self):
        result = _fmt(1024)
        assert "KiB" in result


class TestLogSystemMemory:
    def test_log_system_memory_runs(self):
        """Smoke test: just verify it doesn't crash."""
        import logging
        from graspgenx.utils.compute_utils import log_system_memory

        logger = logging.getLogger("test_compute")
        try:
            log_system_memory(logger, tag="test")
        except FileNotFoundError:
            pytest.skip("/proc/meminfo not available on this platform")


class TestLogSlurm:
    def test_log_slurm_info_runs(self):
        """Smoke test for SLURM logging."""
        import logging
        from graspgenx.utils.compute_utils import log_slurm_info

        logger = logging.getLogger("test_slurm")
        log_slurm_info(logger, tag="test")


class TestLogAllResources:
    def test_log_all_resources_runs(self):
        """Smoke test for combined resource logging."""
        import logging
        from graspgenx.utils.compute_utils import log_all_resources

        logger = logging.getLogger("test_all")
        try:
            log_all_resources(logger, tag="test", include_gpu=False)
        except FileNotFoundError:
            pytest.skip("/proc filesystem not available")
