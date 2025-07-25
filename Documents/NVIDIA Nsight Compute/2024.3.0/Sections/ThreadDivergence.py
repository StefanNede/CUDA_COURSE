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
    MetricRequest("smsp__thread_inst_executed_per_inst_executed.ratio", "thread_inst_executed"),
    MetricRequest("smsp__thread_inst_executed_pred_on_per_inst_executed.ratio", "thread_inst_executed_NPO"),
]


def get_identifier():
    return "ThreadDivergence"

def get_name():
    return "Thread Divergence"

def get_description():
    return "Warp and thread control flow analysis"

def get_section_identifier():
    return "WarpStateStats"

def get_parent_rules_identifiers():
    return ["Compute"]


def get_estimated_speedup(parent_weights, thread_inst_executed, thread_inst_executed_NPO):
    num_threads_used = min(thread_inst_executed, thread_inst_executed_NPO)
    improvement_local = (1 - num_threads_used / 32)

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

    thread_inst_executed = metrics["thread_inst_executed"].value()
    thread_inst_executed_NPO = metrics["thread_inst_executed_NPO"].value()

    fms = []
    threshold = 24

    if thread_inst_executed < threshold or thread_inst_executed_NPO < threshold:
        message = "Instructions are executed in warps, which are groups of 32 threads. Optimal instruction throughput is achieved if all 32 threads of a warp execute the same instruction. The chosen launch configuration, early thread completion, and divergent flow control can significantly lower the number of active threads in a warp per cycle. This kernel achieves an average of {0:.1f} threads being active per cycle.".format(thread_inst_executed)
        fms.append((metrics["thread_inst_executed"].name(), thread_inst_executed, NvRules.IFrontend.Severity_SEVERITY_LOW, "Increase the number of threads per instruction towards 32"))

        if thread_inst_executed_NPO < thread_inst_executed:
            message += " This is further reduced to {0:.1f} threads per warp due to predication. The compiler may use predication to avoid an actual branch. Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads execute the instructions. Try to avoid different execution paths within a warp when possible.".format(thread_inst_executed_NPO)
            fms.append((metrics["thread_inst_executed_NPO"].name(), thread_inst_executed_NPO, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Increase the number of predicated-on threads per instruction towards 32"))

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)

        speedup_type, speedup_value = get_estimated_speedup(parent_weights, thread_inst_executed, thread_inst_executed_NPO)
        fe.speedup(msg_id, speedup_type, speedup_value)

        for fm in fms:
            fe.focus_metric(msg_id, fm[0], fm[1], fm[2], fm[3])
