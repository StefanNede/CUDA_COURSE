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
import sys

def get_identifier():
    return "UncoalescedSharedAccess"

def get_name():
    return "Uncoalesced Shared Accesses"

def get_description():
    return "Uncoalesced Shared Accesses"

def get_section_identifier():
    return "SourceCounters"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    shared_wavefronts_metric_name = "memory_l1_wavefronts_shared"
    shared_wavefronts_metric = action.metric_by_name(shared_wavefronts_metric_name)
    ideal_shared_wavefronts_metric_name = "memory_l1_wavefronts_shared_ideal"
    ideal_shared_wavefronts_metric = action.metric_by_name(ideal_shared_wavefronts_metric_name)
    total_shared_wavefronts = shared_wavefronts_metric.as_uint64()
    total_ideal_shared_wavefronts = ideal_shared_wavefronts_metric.as_uint64()
    # No need to check further if total shared wavefronts match with the ideal value
    if total_shared_wavefronts <= total_ideal_shared_wavefronts:
        return

    num_shared_wavefronts_instances = shared_wavefronts_metric.num_instances()
    num_ideal_shared_wavefronts_instances = ideal_shared_wavefronts_metric.num_instances()
    # We cannot execute the rule if we don't get the same instance count for both metrics
    if num_shared_wavefronts_instances != num_ideal_shared_wavefronts_instances:
        return

    total_diff = 0
    for i in range(num_shared_wavefronts_instances):
        per_instance_shared_wavefronts = shared_wavefronts_metric.as_uint64(i)
        per_instance_ideal_shared_wavefronts = ideal_shared_wavefronts_metric.as_uint64(i)
        if (per_instance_shared_wavefronts != per_instance_ideal_shared_wavefronts):
            total_diff += abs(per_instance_ideal_shared_wavefronts - per_instance_shared_wavefronts)

    if total_diff > 0:
        message = "This kernel has uncoalesced shared accesses resulting in a total of {} excessive wavefronts ({:.0f}% of the total {} wavefronts)." \
            " Check the L1 Wavefronts Shared Excessive table for the primary source locations." \
            " The @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-aa@ has an example on optimizing shared memory accesses." \
            .format(total_diff, 100. * total_diff / total_shared_wavefronts, total_shared_wavefronts)
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)
        fe.focus_metric(msg_id, "derived__memory_l1_wavefronts_shared_excessive", total_diff, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "{} > {}".format(shared_wavefronts_metric_name, ideal_shared_wavefronts_metric_name))
        fe.load_chart_from_file("UncoalescedSharedAccess.chart")
