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
    MetricRequest("smsp__maximum_warps_avg_per_active_cycle", "theoretical_warps"),
    MetricRequest("smsp__warps_active.avg.peak_sustained", "max_warps"),
    MetricRequest("launch__occupancy_limit_blocks"),
    MetricRequest("launch__occupancy_limit_registers"),
    MetricRequest("launch__occupancy_limit_shared_mem"),
    MetricRequest("launch__occupancy_limit_warps"),
]


def get_identifier():
    return "TheoreticalOccupancy"


def get_name():
    return "Theoretical Occupancy"


def get_description():
    return "Analysis of Theoretical Occupancy and its Limiters"


def get_section_identifier():
    return "Occupancy"


def get_parent_rules_identifiers():
    return ["IssueSlotUtilization"]


def get_estimated_speedup(parent_weights, metrics):
    theoretical_warps = metrics["theoretical_warps"].value()
    max_warps = metrics["max_warps"].value()
    improvement_local = 1 - theoretical_warps / max_warps

    parent_speedup_name = "issue_slot_util_speedup_normalized"
    if parent_speedup_name in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        improvement_global = min(parent_weights[parent_speedup_name], improvement_local)
        improvement_percent = improvement_global * 100
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
        improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    parent_weights = fe.receive_dict_from_parent("IssueSlotUtilization")

    theoretical_warps = metrics["theoretical_warps"].value()
    max_warps = metrics["max_warps"].value()

    theoretical_warps_pct_of_peak = (theoretical_warps / max_warps) * 100
    low_theoretical_threshold = 80

    if theoretical_warps_pct_of_peak < low_theoretical_threshold:
        message = "The {:.2f} theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of {}.".format(
            theoretical_warps, int(max_warps)
        )

        limit_types = {
            "blocks": "the number of blocks that can fit on the SM",
            "registers": "the number of required registers",
            "shared_mem": "the required amount of shared memory",
            "warps": "the number of warps within each block",
        }

        limiters = []
        for limiter in limit_types:
            limit_value = metrics[f"launch__occupancy_limit_{limiter}"].value()
            limit_msg = limit_types[limiter]
            limiters.append((limiter, limit_value, limit_msg))

        sorted_limiters = sorted(limiters, key=lambda limit: limit[1])
        last_limiter = -1
        for limiter in sorted_limiters:
            value = limiter[1]
            if last_limiter == -1 or value == last_limiter:
                message += " This kernel's theoretical occupancy ({:.1f}%) is limited by {}.".format(
                    theoretical_warps_pct_of_peak, limiter[2]
                )
                last_limiter = value

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)

        speedup_type, speedup_value = get_estimated_speedup(parent_weights, metrics)
        fe.speedup(msg_id, speedup_type, speedup_value)

        fe.focus_metric(
            msg_id,
            metrics["theoretical_warps"].name(),
            theoretical_warps,
            NvRules.IFrontend.Severity_SEVERITY_HIGH,
            "Increase the theoretical number of warps per schedule that can be issued",
        )
