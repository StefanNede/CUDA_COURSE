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

def get_identifier():
    return "ThreadDivergence"

def get_name():
    return "Thread Divergence"

def get_description():
    return "Warp and thread control flow analysis"

def get_section_identifier():
    return "WarpStateStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    threadInstExecutedName = "smsp__thread_inst_executed_per_inst_executed.ratio"
    threadInstExecutedNPOName = "smsp__thread_inst_executed_pred_on_per_inst_executed.ratio"

    threadInstExecuted = action.metric_by_name(threadInstExecutedName).as_double()
    threadInstExecutedNPO = action.metric_by_name(threadInstExecutedNPOName).as_double()

    fms = []
    threshold = 24

    if threadInstExecuted < threshold or threadInstExecutedNPO < threshold:
        if threadInstExecuted < threshold:
            fms.append((threadInstExecutedName, threadInstExecuted, NvRules.IFrontend.Severity_SEVERITY_LOW, "{} < {}".format(threadInstExecuted, threshold)))
        if threadInstExecutedNPO < threshold:
            fms.append((threadInstExecutedNPOName, threadInstExecutedNPO, NvRules.IFrontend.Severity_SEVERITY_LOW, "{} < {}".format(threadInstExecutedNPO, threshold)))
        message = "Instructions are executed in warps, which are groups of 32 threads. Optimal instruction throughput is achieved if all 32 threads of a warp execute the same instruction. The chosen launch configuration, early thread completion, and divergent flow control can significantly lower the number of active threads in a warp per cycle. This kernel achieves an average of {0:.1f} threads being active per cycle.".format(threadInstExecuted)

        if threadInstExecutedNPO < threadInstExecuted:
            fms.append((threadInstExecutedNPOName, threadInstExecutedNPO, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{} < {}".format(threadInstExecutedNPO, threadInstExecuted)))
            message += " This is further reduced to {0:.1f} threads per warp due to predication. The compiler may use predication to avoid an actual branch. Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads execute the instructions. Try to avoid different execution paths within a warp when possible.".format(threadInstExecutedNPO)
            message += " In addition, ensure your kernel makes use of Independent Thread Scheduling, which allows a warp to reconverge after a data-dependent conditional block by explicitly calling __syncwarp()."

        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)
        for fm in fms:
            fe.focus_metric(msg_id, fm[0], fm[1], fm[2], fm[3])
