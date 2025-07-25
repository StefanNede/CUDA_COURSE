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
    MetricRequest("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained", "inst_executed_ffma_peak"),
    MetricRequest("sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained", "inst_executed_dfma_peak"),
    MetricRequest("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed", "inst_executed_fadd"),
    MetricRequest("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed", "inst_executed_fmul"),
    MetricRequest("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed", "inst_executed_ffma"),
    MetricRequest("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed", "inst_executed_dadd"),
    MetricRequest("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed", "inst_executed_dmul"),
    MetricRequest("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed", "inst_executed_dfma"),
]


def get_identifier():
    return "SOLFPRoofline"

def get_name():
    return "Roofline Analysis"

def get_description():
    return "Floating Point Roofline Analysis"

def get_section_identifier():
    return "SpeedOfLight_RooflineChart"

def get_parent_rules_identifiers():
    return ["HighPipeUtilization"]

def get_estimated_speedup(parent_weights, achieved_fp32, achieved_fp64, peak_fp32, peak_fp64):
    # Estimate the speedup as the 64-bit portion of the compute workload, assuming
    # 32-bit FP pipeline has a higher throughput as 64-bit FP pipeline.
    # To get a global estimate weigh this with the 64-bit FP pipeline utilization
    # (in terms of active cycles).
    if peak_fp64 / peak_fp32 > 1:
        return NvRules.IFrontend.SpeedupType_LOCAL, 0

    improvement_local = (achieved_fp64 / (achieved_fp32 + achieved_fp64)) * (
        1 - peak_fp64 / peak_fp32
    )

    if "fp64_pipeline_utilization_pct" in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        improvement_percent = improvement_local * parent_weights["fp64_pipeline_utilization_pct"]
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

    peak_fp32 = 2 * metrics["inst_executed_ffma_peak"].value()
    peak_fp64 = 2 * metrics["inst_executed_dfma_peak"].value()

    fp32_add_achieved = metrics["inst_executed_fadd"].value()
    fp32_mul_achieved = metrics["inst_executed_fmul"].value()
    fp32_fma_achieved = metrics["inst_executed_ffma"].value()
    achieved_fp32 = fp32_add_achieved + fp32_mul_achieved + 2 * fp32_fma_achieved

    fp64_add_achieved = metrics["inst_executed_dadd"].value()
    fp64_mul_achieved = metrics["inst_executed_dmul"].value()
    fp64_fma_achieved = metrics["inst_executed_dfma"].value()
    achieved_fp64 = fp64_add_achieved + fp64_mul_achieved + 2 * fp64_fma_achieved

    high_utilization_threshold = 0.60
    low_utilization_threshold = 0.15

    achieved_fp64_pct = achieved_fp64 / peak_fp64
    fp64_prefix = "" if achieved_fp64_pct >= 0.01 or achieved_fp64_pct == 0.0 else " close to "
    achieved_fp32_pct = achieved_fp32 / peak_fp32
    fp32_prefix = "" if achieved_fp32_pct >= 0.01 or achieved_fp32_pct == 0.0 else " close to "

    message = "The ratio of peak float (fp32) to double (fp64) performance on this device is {:.0f}:1.".format(peak_fp32 / peak_fp64)
    message += " The kernel achieved {}{:.0f}% of this device's fp32 peak performance and {}{:.0f}% of its fp64 peak performance.".format(fp32_prefix, 100.0 * achieved_fp32_pct, fp64_prefix, 100.0 * achieved_fp64_pct)

    message_profiling_guide = " See the @url:Kernel Profiling Guide:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline@ for more details on roofline analysis."

    if achieved_fp32_pct < high_utilization_threshold and achieved_fp64_pct > low_utilization_threshold:
        message += " If @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ determines that this kernel is fp64 bound, consider using 32-bit precision floating point operations to improve its performance."
        message += message_profiling_guide
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, "FP64/32 Utilization")

        speedup_type, speedup_value = get_estimated_speedup(parent_weights, achieved_fp32, achieved_fp64, peak_fp32, peak_fp64)
        fe.speedup(msg_id, speedup_type, speedup_value)

        if speedup_value > 0:
            fe.focus_metric(msg_id, metrics["inst_executed_dadd"].name(), fp64_add_achieved, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease fp64 ADD instructions")
            fe.focus_metric(msg_id, metrics["inst_executed_dmul"].name(), fp64_mul_achieved, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease fp64 MUL instructions")
            fe.focus_metric(msg_id, metrics["inst_executed_dfma"].name(), fp64_fma_achieved, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Decrease fp64 FMA instructions")
    elif achieved_fp64_pct > high_utilization_threshold and achieved_fp32_pct > high_utilization_threshold:
        message += " If @section:SpeedOfLight:Speed Of Light@ analysis determines that this kernel is compute bound, consider using integer arithmetic instead where applicable."
        message += message_profiling_guide
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, "High FP Utilization")
    else:
        message += message_profiling_guide
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OK, message, "Roofline Analysis")
