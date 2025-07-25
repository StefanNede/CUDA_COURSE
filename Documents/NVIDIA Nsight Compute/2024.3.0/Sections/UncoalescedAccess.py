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
from collections import defaultdict

import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser, Importance

requested_metrics = [
    MetricRequest("memory_l2_theoretical_sectors_global", "l2_sectors"),
    MetricRequest("memory_l2_theoretical_sectors_global_ideal", "l2_sectors_ideal"),
    MetricRequest("derived__memory_l2_theoretical_sectors_global_excessive", "excessive_sectors"),
    MetricRequest("lts__cycles_active.sum", "l2_cycles_active", Importance.OPTIONAL, 0),
    MetricRequest("lts__cycles_elapsed.sum", "l2_cycles_elapsed", Importance.OPTIONAL, 0),
]


def get_identifier():
    return "UncoalescedGlobalAccess"

def get_name():
    return "Uncoalesced Global Accesses"

def get_description():
    return "Uncoalesced Global Accesses"

def get_section_identifier():
    return "SourceCounters"

def get_parent_rules_identifiers():
    return ["Memory"]


def get_estimated_speedup(metrics):
    """Estimate potential speedup from reducing uncoalesced global memory accesses.

    The performance improvement is approximated as relative proportion of excessive
    L2 sectors weighted by time spent in the L2 unit.

    """
    active_cycles = metrics["l2_cycles_active"].value()
    elapsed_cycles = metrics["l2_cycles_elapsed"].value()
    excessive_sectors = metrics["excessive_sectors"].value()
    total_sectors = metrics["l2_sectors"].value()

    if (elapsed_cycles > 0) and (total_sectors > 0):
        improvement_percent = (
            (active_cycles / elapsed_cycles) * (excessive_sectors / total_sectors) * 100
        )
        return NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent
    else:
        return NvRules.IFrontend.SpeedupType_LOCAL, 0


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    l2_sectors_metric = metrics["l2_sectors"]
    l2_sectors_correlation_ids = l2_sectors_metric.correlation_ids()
    ideal_l2_sectors_metric = metrics["l2_sectors_ideal"]
    total_l2_sectors = l2_sectors_metric.value()
    total_ideal_l2_sectors = ideal_l2_sectors_metric.value()
    # No need to check further if total L2 sectors match with the ideal value
    if total_l2_sectors <= total_ideal_l2_sectors:
        return

    num_l2_sectors_instances = l2_sectors_metric.num_instances()
    num_ideal_l2_sectors_instances = ideal_l2_sectors_metric.num_instances()
    # We cannot execute the rule if we don't get the same instance count for both metrics
    if num_l2_sectors_instances != num_ideal_l2_sectors_instances:
        return
    total_diff = 0
    excess_by_line = defaultdict(int)
    total_by_line = defaultdict(int)
    for i in range(num_l2_sectors_instances):
        per_instance_l2_sectors = l2_sectors_metric.as_uint64(i)
        per_instance_ideal_l2_sectors = ideal_l2_sectors_metric.as_uint64(i)
        if (per_instance_l2_sectors != per_instance_ideal_l2_sectors):
            total_diff += abs(per_instance_ideal_l2_sectors - per_instance_l2_sectors)
        # If there are excessive sectors, create source markers in the appropriate places
        if (per_instance_l2_sectors > per_instance_ideal_l2_sectors):
            address = l2_sectors_correlation_ids.as_uint64(i)
            source_info = action.source_info(address)
            excess = abs(per_instance_ideal_l2_sectors - per_instance_l2_sectors)

            # Create source marker in the SASS file
            fe.source_marker("{:.2f}% of this line's global accesses are excessive.".format(excess / per_instance_l2_sectors * 100), address, NvRules.IFrontend.MarkerKind_SASS, NvRules.IFrontend.MsgType_MSG_WARNING)

            # Aggregate diffs per line for the Source file marker
            if source_info is not None:
                line = source_info.line()
                file_name = source_info.file_name()
                excess_by_line[line] += excess
                total_by_line[line] += per_instance_l2_sectors

    for line_number, local_diff in excess_by_line.items():
        # Create source marker in the Source file per affected line
        fe.source_marker("{:.2f}% of this line's global accesses are excessive.".format(local_diff / total_by_line[line_number] * 100), line_number, NvRules.IFrontend.MarkerKind_SOURCE, file_name, NvRules.IFrontend.MsgType_MSG_WARNING)

    if total_diff > 0:
        message = "This kernel has uncoalesced global accesses resulting in a total of {} excessive sectors ({:.0f}% of the total {} sectors)." \
            " Check the L2 Theoretical Sectors Global Excessive table for the primary source locations." \
            " The @url:CUDA Programming Guide:https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses@ has additional information on reducing uncoalesced device memory accesses." \
            .format(total_diff, 100. * total_diff / total_l2_sectors, total_l2_sectors)
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)

        speedup_type, speedup_value = get_estimated_speedup(metrics)
        fe.speedup(msg_id, speedup_type, speedup_value)

        fe.focus_metric(msg_id, metrics["excessive_sectors"].name(), total_diff, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Reduce the number of excessive wavefronts in L2")
        fe.load_chart_from_file("UncoalescedAccess.chart")
