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
    MetricRequest("smsp__sass_inst_executed_op_shared_ld.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("smsp__sass_inst_executed_op_shared_st.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("smsp__inst_executed_op_ldsm.sum", None, Importance.OPTIONAL, 0, False),
    MetricRequest("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", None, Importance.OPTIONAL, 0),
    MetricRequest("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", None, Importance.OPTIONAL, 0),
]


def get_identifier():
    return "SharedMemoryConflicts"

def get_name():
    return "Shared Memory Conflicts"

def get_description():
    return "Detection of shared memory bank conflicts."

def get_section_identifier():
    return "MemoryWorkloadAnalysis_Tables"

def get_parent_rules_identifiers():
    return ["Memory"]


def get_estimated_speedup(parent_weights, bank_conflicts_percent):
    l1tex_throughput_name = "l1tex__throughput.avg.pct_of_peak_sustained_active"

    if l1tex_throughput_name in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        l1tex_throughput = parent_weights[l1tex_throughput_name] / 100
        improvement_percent = bank_conflicts_percent * l1tex_throughput
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
        improvement_percent = bank_conflicts_percent

    return speedup_type, improvement_percent


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    parent_weights = fe.receive_dict_from_parent("Memory")

    shared_access_types = {
        "Shared Load"  : ["mem_shared_op_ld", "shared_ld"],
        "Shared Store" : ["mem_shared_op_st", "shared_st"]
    }

    for access_info, metric_str in shared_access_types.items():
        requests = metrics[f"smsp__sass_inst_executed_op_{metric_str[1]}.sum"].value()
        if access_info == "Shared Load":
            requests += metrics["smsp__inst_executed_op_ldsm.sum"].value()

        if requests == 0:
            continue

        wavefronts = metrics[f"l1tex__data_pipe_lsu_wavefronts_{metric_str[0]}.sum"].value()

        bank_conflicts_metric_name = f"l1tex__data_bank_conflicts_pipe_lsu_{metric_str[0]}.sum"
        bank_conflicts = metrics[bank_conflicts_metric_name].value()

        bank_conflicts_percent = (bank_conflicts * 100.0) / wavefronts if wavefronts > 0 else 0.0
        bank_conflicts_threshold = 10.0

        if (bank_conflicts_percent >= bank_conflicts_threshold):
            message = "The memory access pattern for {}s might not be optimal ".format(access_info.lower())
            message += "and causes on average a {:.1f} - way bank conflict ".format(wavefronts / requests)
            message += "across all {:.0f} {} requests.".format(requests, access_info.lower())
            message += "This results in {:.0f} bank conflicts, ".format(bank_conflicts)
            message += " which represent {:.2f}% ".format(bank_conflicts_percent)
            message += "of the overall {:.0f} wavefronts for {}s.".format(wavefronts, access_info.lower())
            message += " Check the @section:SourceCounters:Source Counters@ section for uncoalesced {}s.".format(access_info.lower())

            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, "{} Bank Conflicts".format(access_info))

            speedup_type, speedup_value = get_estimated_speedup(parent_weights, bank_conflicts_percent)
            fe.speedup(msg_id, speedup_type, speedup_value)

            fe.focus_metric(
                msg_id,
                bank_conflicts_metric_name,
                bank_conflicts,
                NvRules.IFrontend.Severity_SEVERITY_HIGH,
                "Decrease bank conflicts for {}s".format(access_info.lower()),
            )

            l1tex_throughput_name = "l1tex__throughput.avg.pct_of_peak_sustained_active"
            if l1tex_throughput_name in parent_weights:
                fe.focus_metric(
                    msg_id,
                    l1tex_throughput_name,
                    parent_weights[l1tex_throughput_name],
                    NvRules.IFrontend.Severity_SEVERITY_LOW,
                    "The higher the L1/TEX cache throughput the more severe the issue becomes",
                )
