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
import math

import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser

requested_metrics = [
    MetricRequest("launch__waves_per_multiprocessor", "num_waves"),
    MetricRequest("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm_sol_pct"),
    MetricRequest("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "mem_sol_pct"),
    MetricRequest("breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "breakdown_memory"),
]


def get_identifier():
    return "SOLBottleneck"

def get_name():
    return "Bottleneck"

def get_description():
    return "High-level bottleneck detection"

def get_section_identifier():
    return "SpeedOfLight"

def get_max_pipe(breakdown_metrics):
    max_pipe = None
    max_pipe_value = 0

    for name, metric in breakdown_metrics.items():
        pipe_value = metric.value()
        if pipe_value > max_pipe_value:
            max_pipe_value = pipe_value
            max_pipe = name

    if max_pipe:
        tokens = {
            "dram" : "DRAM",
            "l1tex" : "L1",
            "lts" : "L2",
            "ltc" : "L2",
            "fbp" : "DRAM",
            "fbpa" : "DRAM"
        }

        for token in tokens:
            if max_pipe.startswith(token):
                return tokens[token]

    return None


def get_breakdown_metrics(parser, breakdown_metric):
    requests = []
    for metric_name in breakdown_metric.value().split(","):
        request = MetricRequest(metric_name)
        requests.append(request)
    return parser.parse(requests)


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    parser = RequestedMetricsParser(handle, action)
    metrics = parser.parse(requested_metrics)
    breakdown_metrics_memory = get_breakdown_metrics(parser, metrics["breakdown_memory"])

    num_waves_name = metrics["num_waves"].name()
    sm_sol_pct_name = metrics["sm_sol_pct"].name()
    mem_sol_pct_name = metrics["mem_sol_pct"].name()

    num_waves = metrics["num_waves"].value()
    sm_sol_pct = metrics["sm_sol_pct"].value()
    mem_sol_pct = metrics["mem_sol_pct"].value()

    balanced_threshold = 10
    latency_bound_threshold = 60
    no_bound_threshold = 80
    waves_threshold = 1

    msg_type = NvRules.IFrontend.MsgType_MSG_OK

    focus_metrics = []

    if sm_sol_pct >= mem_sol_pct:
        bottleneck_section = "@section:ComputeWorkloadAnalysis:Compute Workload Analysis@"
    else:
        bottleneck_section = "@section:MemoryWorkloadAnalysis:Memory Workload Analysis@"

    if sm_sol_pct < no_bound_threshold and mem_sol_pct < no_bound_threshold:
        if sm_sol_pct < latency_bound_threshold and mem_sol_pct < latency_bound_threshold:
            msg_type = NvRules.IFrontend.MsgType_MSG_OPTIMIZATION
            if num_waves < waves_threshold:
                focus_metrics.append((num_waves_name, num_waves, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{:.3f} < {:.3f}".format(num_waves, waves_threshold)))
                message = "This kernel grid is too small to fill the available resources on this device, resulting in only {:.1f} full waves across all SMs. Look at @section:LaunchStats:Launch Statistics@ for more details.".format(num_waves)
                name = "Small Grid"
            else:
                focus_metrics.append((sm_sol_pct_name, sm_sol_pct, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{:.3f} < {:.3f}".format(sm_sol_pct, no_bound_threshold)))
                focus_metrics.append((mem_sol_pct_name, mem_sol_pct, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{:.3f} < {:.3f}".format(mem_sol_pct, no_bound_threshold)))
                message = "This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below {:.1f}% of peak typically indicate latency issues. Look at @section:SchedulerStats:Scheduler Statistics@ and @section:WarpStateStats:Warp State Statistics@ for potential reasons.".format(latency_bound_threshold)
                name = "Latency Issue"
        elif math.fabs(sm_sol_pct - mem_sol_pct) >= balanced_threshold:
            msg_type = NvRules.IFrontend.MsgType_MSG_OPTIMIZATION
            if sm_sol_pct > mem_sol_pct:
                focus_metrics.append((sm_sol_pct_name, mem_sol_pct, NvRules.IFrontend.Severity_SEVERITY_LOW, "{:.3f} - {:.3f} >= {:.3f}".format(sm_sol_pct, mem_sol_pct, balanced_threshold)))
                message = "Compute is more heavily utilized than Memory: Look at the {} section to see what the compute pipelines are spending their time doing. Also, consider whether any computation is redundant and could be reduced or moved to look-up tables.".format(bottleneck_section)
                name = "High Compute Throughput"
            else:
                focus_metrics.append((mem_sol_pct_name, mem_sol_pct, NvRules.IFrontend.Severity_SEVERITY_LOW, "{:.3f} - {:.3f} >= {:.3f}".format(mem_sol_pct, sm_sol_pct, balanced_threshold)))
                pipe_name = get_max_pipe(breakdown_metrics_memory)
                pipe_msg = "to identify the {} bottleneck".format(pipe_name) if pipe_name else "to see where the memory system bottleneck is"
                message = "Memory is more heavily utilized than Compute: Look at the {} section {}. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute.".format(bottleneck_section, pipe_msg)
                name = "High Memory Throughput"
        else:
            message = "Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ and @section:MemoryWorkloadAnalysis:Memory Workload Analysis@ sections."
            name = "Balanced Throughput"
    else:
        pipe_name = None
        if mem_sol_pct > sm_sol_pct:
            pipe_name = get_max_pipe(breakdown_metrics_memory)
        pipe_msg = pipe_name if pipe_name else "workloads"
        message = "The kernel is utilizing greater than {:.1f}% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing {} in the {} section.".format(no_bound_threshold, pipe_msg, bottleneck_section)
        name = "High Throughput"

    msg_id = fe.message(msg_type, message, name)
    for focus_metric in focus_metrics:
        fe.focus_metric(msg_id, focus_metric[0], focus_metric[1], focus_metric[2], focus_metric[3])

    # Send weights to child rules (with values in [0,1])
    fe.send_dict_to_children(
        {
            "compute_throughput_normalized": sm_sol_pct / 100,
            "memory_throughput_normalized": mem_sol_pct / 100,
            "max_throughput_normalized": max(sm_sol_pct, mem_sol_pct) / 100,
        }
    )
