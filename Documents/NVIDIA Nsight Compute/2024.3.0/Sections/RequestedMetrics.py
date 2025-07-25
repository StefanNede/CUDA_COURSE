from collections import UserDict, namedtuple

import NvRules


# TODO: switch to enum.Enum once this is available in static interpreter
class Importance:
    OPTIONAL = 1
    REQUIRED = 2


_MetricRequest = namedtuple(
    "MetricRequest",
    ["name", "alias", "importance", "default_value", "warn_when_missing"],
    defaults=(None, None, Importance.REQUIRED, 0.0, True),
)


class MetricRequest(_MetricRequest):
    """A metric request containing name, and optional alias, importance and default value.

    Args:
        name (str): The name of the metric.
        alias (str, optional): An alias for the metric name. Defaults to None.
        importance (:obj:`Importance`, optional): Whether the metric is required or
            optional. Defaults to Importance.REQUIRED.
        default_value (int, float, string, optional): A default value for OPTIONAL
            metrics. Defaults to float(0).
        warn_when_missing (bool, optional): Whether to issue a warning when an OPTIONAL
            metric is missing. Defaults to True.
    """

    pass


class MetricNotFoundError(Exception):
    """Exception raised when metric cannot be found.

    Args:
        name (str): The name of the metric that was not found.
        importance (:obj:`Importance`): Whether the metric was marked
            required or optional.
        message (str): A message explaining the error
    """

    def __init__(self, name, importance, message=None):
        importance_str = "Required" if importance == Importance.REQUIRED else "Optional"
        default_message = "{} metric {} could not be found.".format(importance_str, name)

        self.name = name
        self.importance = importance
        self.message = message or default_message
        super().__init__(self.message)


class RequestedMetric:
    """Wrapper class for available (loaded) metrics.

    Args:
        name (str): The name of the metric.
        metric (:obj:`IMetric`): An IMetric object representing the metric.
        importance (:obj:`Importance`, optional): Whether the metric was marked
            required or optional. Defaults to Importance.REQUIRED.
        alias (str, optional): An alias for the metric name. Defaults to None.
    """

    _DEFAULT_IMPORTANCE = Importance.REQUIRED

    def __init__(self, name, metric, importance=None, alias=None):
        self._name = name
        self._metric = metric
        self._importance = importance or RequestedMetric._DEFAULT_IMPORTANCE
        self._alias = alias

    @property
    def name(self):
        return self._name

    @property
    def metric(self):
        return self._metric

    @property
    def importance(self):
        return self._importance

    @property
    def alias(self):
        return self._alias


class RequestedMetricBuilder:
    """Builds a RequestedMetric from an IMetric contained in an IAction.

    Args:
        metrics (:obj:`IAction`): An IAction object (potentially) containing the IMetric
            associated with `name`.
        name (str): The name of the metric.
        importance (:obj:`Importance`, optional): Whether the metric was marked
            required or optional. Defaults to Importance.REQUIRED.
        alias (str, optional): An alias for the metric name. Defaults to None.

    Raises:
        MetricNotFoundError: If no IMetric object associated with `name` is available.
    """

    def __init__(self, metrics, name, importance=Importance.REQUIRED, alias=None):
        self._metric = None
        try:
            metric = metrics[name]
            self._metric = RequestedMetric(name, metric, importance, alias)
        except KeyError:
            raise MetricNotFoundError(name, importance)

    """Returns the RequestedMetric object built."""

    def build(self):
        return self._metric


class RequestedMetricDict(UserDict):
    """Custom dict for IMetric lookup using either the metric's name or alias.

    Allows insertion of RequestedMetric values using a key that matches either
    the metric's name or its alias. Lookup will return the IMetric object directly.
    Lookup can be done using name or alias, independently of how the RequestedMetric
    was inserted.
    """

    def __init__(self):
        self.aliasToName = {}
        super().__init__()

    def __getitem__(self, key):
        try:
            return super().__getitem__(key).metric
        except KeyError:
            pass
        try:
            return super().__getitem__(self.aliasToName[key]).metric
        except KeyError:
            pass
        raise KeyError(key)

    def __setitem__(self, key, item):
        name = item.name
        alias = item.alias

        if (key != name) and (key != alias):
            raise KeyError("Key must match either the metric's name or alias.")

        if alias is not None:
            # check whether alias is already used by another metric
            if (alias in self.aliasToName) and (self.aliasToName[alias] != name):
                raise KeyError("Alias {} is already used by metric {}".format(alias, key))
            # save the alias for later lookup
            self.aliasToName[alias] = name

        return super().__setitem__(name, item)

    def __contains__(self, key):
        is_alias = key in self.aliasToName
        return super().__contains__(key) or is_alias


class RequestedMetricsParser:
    """Convenience class to query IMetric objects from an IAction.

    Args:
        handle: The NvRules ContextHandle.
        action (:obj:`IAction`): IAction object containing the IMetrics to be requested.
    """

    _MISSING_REQUIRED_METRICS_MESSAGE = (
        "Some required metrics are missing; aborted rule execution."
    )

    def __init__(self, handle, action):
        self.handle = handle
        self.frontend = NvRules.get_context(handle).frontend()
        self.metrics = action

    def parse(self, requested_metrics):
        """Parse a list of `MetricRequest`s and return a custom dict of `IMetric` objects.

        Args:
            requested_metrics (List[MetricRequest]): A list of requested metrics.

        Returns:
            A RequestedMetricDict, which returns an IMetric object for any valid metric
                name or alias.

        Raises:
            SystemError: If any REQUIRED metric is not contained in the IAction object.
        """
        parsed_metrics = RequestedMetricDict()
        found_missing_required_metrics = False
        found_missing_optional_metrics = False

        for metric in requested_metrics:
            try:
                parsed_metrics[metric.name] = RequestedMetricBuilder(
                    name=metric.name,
                    metrics=self.metrics,
                    importance=metric.importance,
                    alias=metric.alias,
                ).build()
            except MetricNotFoundError as error:
                if error.importance == Importance.OPTIONAL:
                    parsed_metrics[metric.name] = RequestedMetric(
                        name=metric.name,
                        metric=self._create_fallback_metric(metric),
                        importance=metric.importance,
                        alias=metric.alias,
                    )
                    # issue a warning for the first missing optional metric (by default)
                    if metric.warn_when_missing and not found_missing_optional_metrics:
                        self.frontend.message(
                            NvRules.IFrontend.MsgType_MSG_WARNING,
                            self._get_missing_optional_metric_warning(error),
                        )
                        found_missing_optional_metrics = True
                elif error.importance == Importance.REQUIRED:
                    found_missing_required_metrics = True
                    self.frontend.message(
                        NvRules.IFrontend.MsgType_MSG_ERROR,
                        self._get_missing_required_metric_warning(error),
                    )

        if found_missing_required_metrics:
            NvRules.raise_exception(
                self.handle, RequestedMetricsParser._MISSING_REQUIRED_METRICS_MESSAGE
            )

        return parsed_metrics

    def _create_fallback_metric(self, metric_request):
        if metric_request.default_value is None:
            return None

        value = metric_request.default_value
        fallback_metric = self.metrics.add_metric(metric_request.name)

        if isinstance(value, int) and value >= 0:
            fallback_metric.set_uint64(NvRules.IMetric.ValueKind_UINT64, value)
        elif isinstance(value, float):
            fallback_metric.set_double(NvRules.IMetric.ValueKind_DOUBLE, value)
        elif isinstance(value, str):
            fallback_metric.set_string(NvRules.IMetric.ValueKind_STRING, value)
        else:
            raise ValueError("Can only create fallback metric from uint, float or str.")

        return fallback_metric

    def _get_missing_optional_metric_warning(self, error):
        return str(
            "The optional metric {} could not be found. "
            "Collecting it as an additional metric could enable the rule "
            "to provide more guidance.".format(error.name)
        )

    def _get_missing_required_metric_warning(self, error):
        return error.message
