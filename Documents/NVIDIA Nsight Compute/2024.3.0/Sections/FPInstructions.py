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
    MetricRequest("sass__inst_executed_per_opcode"),
]


def get_identifier():
    return "FPInstructions"

def get_name():
    return "FP32/64 Instructions"

def get_description():
    return "Floating-point instruction analysis."

def get_section_identifier():
    return "InstructionStats"

def get_parent_rules_identifiers():
    return ["HighPipeUtilization"]

def get_estimated_speedup(pipeline_utilization_pct, fused_instructions, non_fused_instructions):
    # To calculate the speedup, assume we can convert non-fused to fused instructions,
    # which have double the throughput.
    # To get a global estimate weigh this with the FP pipeline utilization
    # (in terms of active cycles).
    all_instructions = non_fused_instructions + fused_instructions
    improvement_local = 0.5 * (non_fused_instructions / all_instructions)

    if pipeline_utilization_pct is not None:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        improvement_percent = improvement_local * pipeline_utilization_pct
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
        improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    parent_weights = fe.receive_dict_from_parent("HighPipeUtilization")

    fp_types = {
        32 : [ "FADD", "FMUL", "FFMA" ],
        64 : [ "DADD", "DMUL", "DFMA" ]
    }

    # the correlation IDs of sass__inst_executed_per_opcode are the opcode mnemonics
    inst_per_opcode = metrics["sass__inst_executed_per_opcode"]
    num_opcodes = inst_per_opcode.num_instances()
    opcodes = inst_per_opcode.correlation_ids()

    # analyze both 32 and 64 bit
    for fp_type in fp_types:
        fp_insts = dict()
        fp_opcodes = fp_types[fp_type]
        # get number of instructions by opcode
        for i in range(0,num_opcodes):
            op = opcodes.as_string(i).upper()
            if op in fp_opcodes:
                fp_insts[op] = inst_per_opcode.as_uint64(i)

        # calculate the sum of low- and high-throughput instructions
        non_fused = 0
        for i in range(0, 2):
            op = fp_opcodes[i]
            if op in fp_insts:
                non_fused += fp_insts[op]

        fused = 0
        op = fp_opcodes[2]
        if op in fp_insts:
            fused += fp_insts[op]

        if non_fused > 0 or fused > 0:
            # high-throughput/fused instructions have twice the throughput of non-fused ones
            ratio = (non_fused / (non_fused + fused)) / 2
            if ratio > 0.1:
                message = "This kernel executes {} fused and {} non-fused FP{} instructions.".format(fused, non_fused, fp_type)
                message += " By converting pairs of non-fused instructions to their @url:fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP{} performance could be increased by up to {:.0f}%"\
                    " (relative to its current performance)."\
                    " Check the Source page to identify where this kernel executes FP{} instructions.".format(fp_type, 100. * ratio, fp_type)
                message_title = "FP{} Non-Fused Instructions".format(fp_type)
                msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, message_title)

                pipeline_utilization_pct = None
                parent_weight_name = "fp{}_pipeline_utilization_pct".format(fp_type)
                if parent_weight_name in parent_weights:
                    pipeline_utilization_pct = parent_weights[parent_weight_name]
                speedup_type, speedup_value = get_estimated_speedup(pipeline_utilization_pct, fused, non_fused)
                fe.speedup(msg_id, speedup_type, speedup_value)

                fe.focus_metric(msg_id, "sass__inst_executed_per_opcode", non_fused, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease the number of non-fused floating-point instructions (FADD, FMUL, DADD, DMUL)")
                if pipeline_utilization_pct is not None:
                    if fp_type == 32:
                        metric_name = "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"
                    else:
                        metric_name = "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active"
                    fe.focus_metric(msg_id, metric_name, pipeline_utilization_pct, NvRules.IFrontend.Severity_SEVERITY_LOW, "The higher the utilization of the pipeline the more severe the issue becomes")
