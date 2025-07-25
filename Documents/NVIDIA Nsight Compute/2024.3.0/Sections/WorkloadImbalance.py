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

requested_metrics = [
    MetricRequest("dram__cycles_active.avg", None, Importance.OPTIONAL, 0),
    MetricRequest("dram__cycles_active.max", None, Importance.OPTIONAL, 0),
    MetricRequest("dram__cycles_active.min", None, Importance.OPTIONAL, 0),
    MetricRequest("dram__cycles_active.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("dram__cycles_elapsed.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__cycles_active.avg", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__cycles_active.max", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__cycles_active.min", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__cycles_active.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__cycles_elapsed.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("lts__cycles_active.avg", None, Importance.OPTIONAL, 0),
    MetricRequest("lts__cycles_active.max", None, Importance.OPTIONAL, 0),
    MetricRequest("lts__cycles_active.min", None, Importance.OPTIONAL, 0),
    MetricRequest("lts__cycles_active.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("lts__cycles_elapsed.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("sm__cycles_active.avg", None, Importance.OPTIONAL, 0),
    MetricRequest("sm__cycles_active.max", None, Importance.OPTIONAL, 0),
    MetricRequest("sm__cycles_active.min", None, Importance.OPTIONAL, 0),
    MetricRequest("sm__cycles_active.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("sm__cycles_elapsed.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("smsp__cycles_active.avg", None, Importance.OPTIONAL, 0),
    MetricRequest("smsp__cycles_active.max", None, Importance.OPTIONAL, 0),
    MetricRequest("smsp__cycles_active.min", None, Importance.OPTIONAL, 0),
    MetricRequest("smsp__cycles_active.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("smsp__cycles_elapsed.sum", None, Importance.OPTIONAL, 0),
]


def get_identifier():
    return "WorkloadImbalance"


def get_name():
    return "Workload Imbalance"


def get_description():
    return "Analysis of workload distribution in active cycles of SM, SMP, SMSP, L1 & L2 caches, and DRAM"


def get_section_identifier():
    return "WorkloadDistribution"


def get_parent_rules_identifiers():
    return ["Compute"]

def analyze_imbalance(fe, metrics, metric_base_name, id, min_speedup, recommendation=None):

    # Metrics where unavailable
    if (metrics[f'{metric_base_name}.max'].value() == 0 or metrics[f'{metric_base_name}.avg'].value() == 0):
        return

    max_distance_from_avg = (1 - (metrics[f'{metric_base_name}.avg'].value() / metrics[f'{metric_base_name}.max'].value())) * 100
    min_distance_from_avg = (1 - (metrics[f'{metric_base_name}.min'].value() / metrics[f'{metric_base_name}.avg'].value())) * 100

    # In the case, for example, where you have multiple SM's doing "much more work" than the rest of the SM,
    # that means that their load can distributed over the remaining SM's achieving that difference percentage as a speed up.
    #
    # For example, if we have 4 SMs, with the following active cycles [1:100, 2:100, 3:100, 4:180].
    # In this case SM4 is the bottleneck. The sum of all active cycles is 480. The average of this is: 120
    # That means we have a max difference of = (1 - (120/180)) * 100 = 33%.
    #
    # Distributing the load we would have: [1:120, 2:120, 3:120, 4: 120] which
    # lowers the number of active cycles of the bottle neck SM by 33% (i.e speeding up by 33%)

    total_elapsed = metrics[f'{metric_base_name.replace("active", "elapsed")}.sum'].value()
    total_active = metrics[f'{metric_base_name}.sum'].value()
    speedup = max_distance_from_avg * (total_active / total_elapsed)

    # Speedup is less than the minimum required for reporting
    if speedup < min_speedup:
        return

    if (max_distance_from_avg >= min_speedup
        and min_distance_from_avg >= min_speedup
        and abs(max_distance_from_avg - min_distance_from_avg) < 2):
        # If max and min distance are less than 2% away from  each other and both are above min speedup
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION,
                f"One or more {id}s have a much higher number of active cycles than the average number of active cycles. Additionally, "
                f"other {id}s have a much lower number of active cycles than the average number of active cycles. "
                f"Maximum instance value is {max_distance_from_avg:.2f}% above the average, while the minimum instance value is {min_distance_from_avg:.2f}% below the average.",
                f"{id}s Workload Imbalance")
    elif (max_distance_from_avg >= min_speedup
          and max_distance_from_avg > min_distance_from_avg):
        # Max load is the major contributor to the imbalance (i.e there is room to distrubute work to other parts)
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION,
                f"One or more {id}s have a much higher number of active cycles than the average number of active cycles. "
                f"Maximum instance value is {max_distance_from_avg:.2f}% above the average, while the minimum instance value is {min_distance_from_avg:.2f}% below the average.",
                f"{id}s Workload Imbalance")
    elif (min_distance_from_avg >= min_speedup
          and min_distance_from_avg > max_distance_from_avg):
        # Min load is the major contributor to the imbalance (i.e assign more to this part)
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION,
                f"One or more {id}s have a much lower number of active cycles than the average number of active cycles. "
                f"Maximum instance value is {max_distance_from_avg:.2f}% above the average, while the minimum instance value is {min_distance_from_avg:.2f}% below the average.",
                f"{id}s Workload Imbalance")
    else:
        return

    fe.speedup(msg_id, NvRules.IFrontend.SpeedupType_GLOBAL, speedup)
    fe.focus_metric(
        msg_id,
        metrics[f"{metric_base_name}.avg"].name(),
        metrics[f"{metric_base_name}.avg"].value(),
        NvRules.IFrontend.Severity_SEVERITY_HIGH,
        (recommendation if recommendation is not None else f"Balancing the number of active cycles across {id}s would result in a more optimized kernel")
    )

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    analyze_imbalance(fe=fe, metrics=metrics, metric_base_name="sm__cycles_active", id="SM", min_speedup=5)
    analyze_imbalance(fe=fe, metrics=metrics, metric_base_name="smsp__cycles_active", id="SMSP", min_speedup=5)
    analyze_imbalance(fe=fe, metrics=metrics, metric_base_name="l1tex__cycles_active", id="L1 Slice", min_speedup=5)
    analyze_imbalance(fe=fe, metrics=metrics, metric_base_name="lts__cycles_active", id="L2 Slice", min_speedup=5)
    analyze_imbalance(fe=fe, metrics=metrics, metric_base_name="dram__cycles_active", id="DRAM Slice", min_speedup=5)