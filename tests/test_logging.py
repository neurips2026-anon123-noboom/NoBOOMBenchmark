import logging

from noboom_benchmark.noboom_lib.core.logging import setup_logging


def test_setup_logging_keeps_noisy_dependency_loggers_quiet() -> None:
    setup_logging(verbosity=1)

    assert logging.getLogger("botocore").level == logging.WARNING
    assert logging.getLogger("s3transfer").level == logging.WARNING
    assert logging.getLogger("urllib3.connection").level == logging.ERROR
