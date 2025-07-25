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
    MetricRequest("smsp__issue_active.avg.per_cycle_active", "issue_active"),
    MetricRequest("smsp__maximum_warps_avg_per_active_cycle", "theoretical_warps"),
    MetricRequest("smsp__warps_active.avg.per_cycle_active", "active_warps"),
    MetricRequest("smsp__warps_eligible.avg.per_cycle_active", "eligible_warps"),
    MetricRequest("smsp__warps_active.avg.peak_sustained", "max_warps"),
]


def get_identifier():
    return "IssueSlotUtilization"

def get_name():
    return "Issue Slot Utilization"

def get_description():
    return "Scheduler instruction issue analysis"

def get_section_identifier():
    return "SchedulerStats"

def get_parent_rules_identifiers():
    return ["Compute"]


def get_estimated_speedup(parent_weights, metrics):
    issue_active = metrics["issue_active"].value()
    improvement_local = 1 - issue_active

    throughput_name = "max_throughput_normalized"
    if throughput_name in parent_weights:
        upper_bound = 1 - parent_weights[throughput_name]
        improvement_local = min(improvement_local, upper_bound)

    speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
    improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    parent_weights = fe.receive_dict_from_parent("Compute")

    issue_active = metrics["issue_active"].value()
    theoretical_warps = metrics["theoretical_warps"].value()
    active_warps = metrics["active_warps"].value()
    eligible_warps = metrics["eligible_warps"].value()
    max_warps = metrics["max_warps"].value()

    issueActiveTarget = 0.6

    if issue_active < issueActiveTarget:
        message = "Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only issues an instruction every {:.1f} cycles. This might leave hardware resources underutilized and may lead to less optimal performance.".format(1./issue_active)
        message += " Out of the maximum of {} warps per scheduler, this kernel allocates an average of {:.2f} active warps per scheduler,".format(int(max_warps), active_warps)

        if active_warps < 1.0:
            message += " which already limits the scheduler to less than a warp per instruction."
        else:
            message += " but only an average of {:.2f} warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused.".format(eligible_warps)
            if active_warps / theoretical_warps < 0.8:
                message += " To increase the number of eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons on the "
                message += '@section:WarpStateStats:Warp State Statistics@ and @section:SourceCounters:Source Counters@ sections.'
            else:
                message += " To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp."
                message += ' Reducing stalls indicated on the @section:WarpStateStats:Warp State Statistics@ and @section:SourceCounters:Source Counters@ sections can help, too.'

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)

        speedup_type, speedup_value = get_estimated_speedup(parent_weights, metrics)
        fe.speedup(msg_id, speedup_type, speedup_value)
        parent_weights = {
            **parent_weights,
            "issue_slot_util_speedup_normalized": speedup_value / 100,
        }

        fe.focus_metric(msg_id, metrics["issue_active"].name(), issue_active, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Increase the average number of instructions issued per cycle")

    fe.send_dict_to_children(parent_weights)
