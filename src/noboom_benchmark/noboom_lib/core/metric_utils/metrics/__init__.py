from noboom.tsad.metrics import edf, ldf

from .alarm import alarm_score
from .average_alarm_frequency import aaf
from .event_recall import event_recall


def is_metric_binary(metric_name: str):
    """Check whether a metric is binary-classification based.

    Args:
        metric_name (str): Metric name to check.

    Returns:
        bool: True if the metric expects binary labels.
    """
    return metric_name in ["alarm_score", "aaf", "event_recall"]


def get_metric_by_name(metric_name: str):
    """Get a metric function by its name.

    Args:
        metric_name (str): Metric name to resolve.

    Returns:
        Callable: Metric function.

    Raises:
        ValueError: If the metric name is unknown.
    """
    if metric_name == "alarm_score":
        return alarm_score
    elif metric_name == "aaf":
        return aaf
    elif metric_name == "event_recall":
        return event_recall
    elif metric_name == "edf":
        return edf
    elif metric_name == "ldf":
        return ldf
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
