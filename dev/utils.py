import os
import argparse
from datetime import datetime
from pathlib import Path
import logging

def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter
        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False
    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_logger(exp_dir, log_prefix, log_level='info', use_console=True):
    """
    Log setup
    Args:
        exp_dir:
        log_prefix:
        log_level: "debug", "info", "warning", "error", "critical"
        use_console: if True, will also print logs to console
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = Path(exp_dir) / 'log'
    # if not log_path.exists():
    log_path.mkdir(parents=True, exist_ok=True)

    # formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    formatter = '%(asctime)s - %(levelname)s: %(message)s'
    log_filename = os.fspath(log_path / f"{log_prefix}-{date_time}.log")

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
    )

    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter, datefmt='%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(console)
