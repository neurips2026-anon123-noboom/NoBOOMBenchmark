import logging
import sys


def setup_logging(verbosity: int = 0) -> None:
    """Configure the root logger to emit logs to stdout.

    Args:
        verbosity (int): Verbosity level; values > 0 enable DEBUG. Defaults to 0.

    Returns:
        None: Logging is configured for the current process.

    Side Effects:
        Resets the root logging configuration with ``logging.basicConfig``.
    """
    level = logging.INFO if verbosity <= 0 else logging.DEBUG

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,   # ensure logs go to stdout
        force=True,          # override any previous logging config
    )
    for logger_name in ("boto3", "botocore", "s3transfer", "urllib3.connectionpool"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.getLogger("urllib3.connection").setLevel(logging.ERROR)
