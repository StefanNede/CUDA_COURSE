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
    return "UncoalescedGlobalAccess"

def get_name():
    return "Uncoalesced Global Accesses"

def get_description():
    return "Uncoalesced Global Accesses"

def get_section_identifier():
    return "SourceCounters"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    l2_sectors_metric_name = "memory_l2_theoretical_sectors_global"
    l2_sectors_metric = action.metric_by_name(l2_sectors_metric_name)
    ideal_l2_sectors_metric_name = "memory_l2_theoretical_sectors_global_ideal"
    ideal_l2_sectors_metric = action.metric_by_name(ideal_l2_sectors_metric_name)
    total_l2_sectors = l2_sectors_metric.as_uint64()
    total_ideal_l2_sectors = ideal_l2_sectors_metric.as_uint64()
    # No need to check further if total L2 sectors match with the ideal value
    if total_l2_sectors <= total_ideal_l2_sectors:
        return

    num_l2_sectors_instances = l2_sectors_metric.num_instances()
    num_ideal_l2_sectors_instances = ideal_l2_sectors_metric.num_instances()
    # We cannot execute the rule if we don't get the same instance count for both metrics
    if num_l2_sectors_instances != num_ideal_l2_sectors_instances:
        return

    total_diff = 0
    for i in range(num_l2_sectors_instances):
        per_instance_l2_sectors = l2_sectors_metric.as_uint64(i)
        per_instance_ideal_l2_sectors = ideal_l2_sectors_metric.as_uint64(i)
        if (per_instance_l2_sectors != per_instance_ideal_l2_sectors):
            total_diff += abs(per_instance_ideal_l2_sectors - per_instance_l2_sectors)

    if total_diff > 0:
        message = "This kernel has uncoalesced global accesses resulting in a total of {} excessive sectors ({:.0f}% of the total {} sectors)." \
            " Check the L2 Theoretical Sectors Global Excessive table for the primary source locations." \
            " The @url:CUDA Programming Guide:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses@ had additional information on reducing uncoalesced device memory accesses." \
            .format(total_diff, 100. * total_diff / total_l2_sectors, total_l2_sectors)
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)
        fe.focus_metric(msg_id, "derived__memory_l2_theoretical_sectors_global_excessive", total_diff, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "{} > {}".format(l2_sectors_metric_name, ideal_l2_sectors_metric_name))
        fe.load_chart_from_file("UncoalescedAccess.chart")
