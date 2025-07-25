# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import math

def get_identifier():
    return "SOLFPRoofline"

def get_name():
    return "Roofline Analysis"

def get_description():
    return "Floating Point Roofline Analysis"

def get_section_identifier():
    return "SpeedOfLight_RooflineChart"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    peak_fp32 = 2 * action.metric_by_name("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained").as_double()
    peak_fp64 = 2 * action.metric_by_name("sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained").as_double()

    achieved_fp32 = 0.0
    achieved_fp32 += action.metric_by_name("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp32 += action.metric_by_name("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp32 += 2 * action.metric_by_name("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed").as_double()

    achieved_fp64 = 0.0
    achieved_fp64 += action.metric_by_name("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp64 += action.metric_by_name("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed").as_double()
    achieved_fp64 += 2 * action.metric_by_name("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed").as_double()

    msg_type = NvRules.IFrontend.MsgType_MSG_OK

    ratio = peak_fp32 / peak_fp64
    message = "The ratio of peak float (fp32) to double (fp64) performance on this device is {:.0f}:1.".format(ratio)

    high_utilization_threshold = 0.60
    low_utilization_threshold = 0.15

    achieved_fp64_pct = achieved_fp64 / peak_fp64
    fp64_prefix = "" if achieved_fp64_pct >= 0.01 or achieved_fp64_pct == 0.0 else " close to "
    achieved_fp32_pct = achieved_fp32 / peak_fp32
    fp32_prefix = "" if achieved_fp32_pct >= 0.01 or achieved_fp32_pct == 0.0 else " close to "

    message += " The kernel achieved {}{:.0f}% of this device's fp32 peak performance and {}{:.0f}% of its fp64 peak performance.".format(fp32_prefix, 100.0 * achieved_fp32_pct, fp64_prefix, 100.0 * achieved_fp64_pct)
    msg_name = "Roofline Analysis"

    if achieved_fp32_pct < high_utilization_threshold and achieved_fp64_pct > low_utilization_threshold:
        msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
        message += " If @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ determines that this kernel is fp64 bound, consider using 32-bit precision floating point operations to improve its performance."
        msg_name = "FP64/32 Utilization"
    elif achieved_fp64_pct > high_utilization_threshold and achieved_fp32_pct > high_utilization_threshold:
        msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
        message += " If @section:SpeedOfLight:Speed Of Light@ analysis determines that this kernel is compute bound, consider using integer arithmetic instead where applicable."
        msg_name = "High FP Utilization"

    message += " See the @url:Kernel Profiling Guide:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline@ for more details on roofline analysis."

    fe.message(msg_type, message, msg_name)

    if achieved_fp64_pct > 0.1:
        all_metrics = action.metric_names()
        fp64_pipe_name = "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed"
        if fp64_pipe_name in all_metrics:
            fp64_pipe = action.metric_by_name(fp64_pipe_name).as_double()
            diff_pct = fp64_pipe - 100.0 * achieved_fp64_pct
            threshold = 10
            if diff_pct > threshold:
                message = "The achieved fp64 performance is {:.0f}% lower than the fp64 pipeline utilization." \
                    " Check the @section:InstructionStats:Instruction Statistics@ section to see if using fused instructions can benefit this kernel." \
                    .format(diff_pct)
                msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message, msg_name)

                sm_tp_name = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
                if (sm_tp_name in all_metrics) and (action.metric_by_name(sm_tp_name).as_double() - fp64_pipe < 10):
                    severity = NvRules.IFrontend.Severity_SEVERITY_HIGH
                else:
                    severity = NvRules.IFrontend.Severity_SEVERITY_LOW
                fe.focus_metric(msg_id, fp64_pipe_name, fp64_pipe, severity, "{} - achieved_fp64 > {}".format(fp64_pipe_name, threshold))
