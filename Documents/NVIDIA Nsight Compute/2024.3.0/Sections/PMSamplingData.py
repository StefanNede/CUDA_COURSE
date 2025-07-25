# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser, Importance

requested_metrics_base = [
    MetricRequest("device__attribute_compute_capability_major", "cc_major"),
    MetricRequest("device__attribute_compute_capability_minor", "cc_minor"),
]

requested_metrics_cycles = [
    MetricRequest("profiler__pmsampler_interval_cycles", "interval", Importance.OPTIONAL, 0, False),
    MetricRequest("gpc__cycles_elapsed.max", "duration", Importance.OPTIONAL, 0, False),
]

requested_metrics_time = [
    MetricRequest("profiler__pmsampler_interval_time", "interval", Importance.OPTIONAL, 0, False),
    MetricRequest("gpu__time_duration.sum", "duration", Importance.OPTIONAL, 0, False),
]

def get_identifier():
    return "PMSamplingData"


def get_name():
    return "PM Sampling Data"


def get_description():
    return "Detection of PM sampling data collection issues"


def get_section_identifier():
    return "PmSampling"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    metrics_base = RequestedMetricsParser(handle, action).parse(requested_metrics_base)
    cc = metrics_base["cc_major"].value() * 10 + metrics_base["cc_minor"].value()
    if cc < 75:
        # PM sampling is supported starting with SM 7.5
        return

    if cc > 80:
        metrics = RequestedMetricsParser(handle, action).parse(requested_metrics_time)
        min_interval = 1000
    else:
        metrics = RequestedMetricsParser(handle, action).parse(requested_metrics_cycles)
        min_interval = 20000

    sampling_interval = metrics['interval'].value()
    sampling_duration = metrics['duration'].value()

    if sampling_duration and sampling_interval:
        ratio = sampling_interval / sampling_duration
        message = ""
        if ratio >= 1:
            message = "Sampling interval is {:.1f}x of the workload duration, which likely results in no or very few collected samples.".format(ratio)
        elif ratio > 0.1:
            message = "Sampling interval is larger than 10% of the workload duration, which likely results in very few collected samples.".format(ratio)

        if message:
            if sampling_interval > min_interval:
                message += " For better results, use the --pm-sampling-interval option to reduce the sampling interval."
                message += " Use --pm-sampling-buffer-size to increase the sampling buffer size for the smaller interval, or don't set a fixed buffer size and let the tool adjust it automatically."
            fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)
