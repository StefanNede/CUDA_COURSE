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
from RequestedMetrics import MetricRequest, RequestedMetricsParser

requested_metrics = [
    MetricRequest("sm__instruction_throughput.avg.pct_of_peak_sustained_active", "instruction_throughput"),
    MetricRequest("sm__inst_issued.avg.pct_of_peak_sustained_active", "inst_issued_avg"),
    MetricRequest("sm__inst_issued.max.pct_of_peak_sustained_active", "inst_issued_max"),
]


def get_identifier():
    return "SlowPipeLimiter"

def get_name():
    return "Slow Pipe Limiter"

def get_description():
    return "Slow pipe limiting compute utilization"

def get_section_identifier():
    return "ComputeWorkloadAnalysis"


def get_parent_rules_identifiers():
    return ["Compute"]


def get_estimated_speedup(parent_weights, metrics):
    """Estimate potential speedup from decreasing the usage of slow pipes.

    The performance improvement is approximated as the relative part of instructions
    not issued.
    In case the compute (SM) throughput was collected,
    the above approximation can be improved by weighing it with the achieved throughput.

    """
    inst_issued_avg = metrics["inst_issued_avg"].value()
    inst_issued_max = metrics["inst_issued_max"].value()
    improvement_local = (inst_issued_max - inst_issued_avg) / inst_issued_max

    compute_throughput_name = "compute_throughput_normalized"
    if compute_throughput_name in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        improvement_percent = improvement_local * parent_weights[compute_throughput_name] * 100
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
        improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    parent_weights = fe.receive_dict_from_parent("Compute")

    sm_busy = metrics["instruction_throughput"].value()
    inst_issued_avg = metrics["inst_issued_avg"].value()
    inst_issued_max = metrics["inst_issued_max"].value()

    no_bound_threshold = 80
    issued_avg_threshold = 20
    diff_threshold = 25

    doc_msg = " See the @url:Kernel Profiling Guide:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder@ for the workloads handled by each pipeline."

    pipe_diff = inst_issued_max - inst_issued_avg
    if sm_busy >= no_bound_threshold and inst_issued_avg < issued_avg_threshold and pipe_diff > diff_threshold:
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION,
        "It is possible that a slow pipeline is preventing better kernel performance."\
        " The average pipeline utilization of {:.1f}% is {:.1f}% lower than the maximum utilization of {:.1f}%."\
        " Try moving compute to other pipelines, e.g. from fp64 to fp32 or int."\
        "{}".format(inst_issued_avg, pipe_diff, inst_issued_max, doc_msg), "Slow Pipeline")

        speedup_type, speedup_value = get_estimated_speedup(parent_weights, metrics)
        fe.speedup(msg_id, speedup_type, speedup_value)

        fe.focus_metric(msg_id, metrics["instruction_throughput"].name(), sm_busy, NvRules.IFrontend.Severity_SEVERITY_LOW,
            "The higher the instruction throughput the more likely is the impact of a slow pipeline")
        fe.focus_metric(msg_id, metrics["inst_issued_avg"].name(), inst_issued_avg, NvRules.IFrontend.Severity_SEVERITY_HIGH,
            "Increase the average pipeline utilization towards the maximum utilization ({:.1f}%)".format(inst_issued_max))
