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
    MetricRequest("memory_l1_wavefronts_shared", "shared_wavefronts"),
    MetricRequest("memory_l1_wavefronts_shared_ideal", "shared_wavefronts_ideal"),
    MetricRequest("derived__memory_l1_wavefronts_shared_excessive", "excessive_wavefronts"),
    MetricRequest("l1tex__cycles_active.sum", "l1tex_cycles_active", Importance.OPTIONAL, 0),
    MetricRequest("l1tex__cycles_elapsed.sum", "l1tex_cycles_elapsed", Importance.OPTIONAL, 0),
]

def get_identifier():
    return "UncoalescedSharedAccess"

def get_name():
    return "Uncoalesced Shared Accesses"

def get_description():
    return "Uncoalesced Shared Accesses"

def get_section_identifier():
    return "SourceCounters"

def get_parent_rules_identifiers():
    return ["Memory"]


def get_estimated_speedup(metrics):
    """Estimate potential speedup from reducing uncoalesced shared memory accesses.

    The performance improvement is approximated as relative proportion of excessive
    wavefronts weighted by time spent in the L1TEX unit.

    """
    active_cycles = metrics["l1tex_cycles_active"].value()
    elapsed_cycles = metrics["l1tex_cycles_elapsed"].value()
    excessive_wavefronts = metrics["excessive_wavefronts"].value()
    total_wavefronts = metrics["shared_wavefronts"].value()

    if (elapsed_cycles > 0) and (total_wavefronts > 0):
        improvement_percent = (
            (active_cycles / elapsed_cycles)
            * (excessive_wavefronts / total_wavefronts)
            * 100
        )
        return NvRules.IFrontend.SpeedupType_GLOBAL, improvement_percent
    else:
        return NvRules.IFrontend.SpeedupType_LOCAL, 0


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    shared_wavefronts_metric = metrics["shared_wavefronts"]
    shared_wavefronts_correlation_ids = shared_wavefronts_metric.correlation_ids()
    ideal_shared_wavefronts_metric = metrics["shared_wavefronts_ideal"]
    total_shared_wavefronts = shared_wavefronts_metric.value()
    total_ideal_shared_wavefronts = ideal_shared_wavefronts_metric.value()
    # No need to check further if total shared wavefronts match with the ideal value
    if total_shared_wavefronts <= total_ideal_shared_wavefronts:
        return

    num_shared_wavefronts_instances = shared_wavefronts_metric.num_instances()
    num_ideal_shared_wavefronts_instances = ideal_shared_wavefronts_metric.num_instances()
    # We cannot execute the rule if we don't get the same instance count for both metrics
    if num_shared_wavefronts_instances != num_ideal_shared_wavefronts_instances:
        return

    total_diff = 0
    excess_by_line = defaultdict(int)
    total_by_line = defaultdict(int)
    for i in range(num_shared_wavefronts_instances):
        per_instance_shared_wavefronts = shared_wavefronts_metric.as_uint64(i)
        per_instance_ideal_shared_wavefronts = ideal_shared_wavefronts_metric.as_uint64(i)
        if (per_instance_shared_wavefronts != per_instance_ideal_shared_wavefronts):
            total_diff += abs(per_instance_ideal_shared_wavefronts - per_instance_shared_wavefronts)
        # If there are excessive wavefronts, create source markers in the appropriate places
        if (per_instance_shared_wavefronts > per_instance_ideal_shared_wavefronts):
            address = shared_wavefronts_correlation_ids.as_uint64(i)
            source_info = action.source_info(address)
            excess = abs(per_instance_ideal_shared_wavefronts - per_instance_shared_wavefronts)

            # Create source marker in the SASS file
            fe.source_marker("{:.2f}% of this line's shared wavefronts are excessive.".format(excess / per_instance_shared_wavefronts * 100), address, NvRules.IFrontend.MarkerKind_SASS, NvRules.IFrontend.MsgType_MSG_WARNING)

            # Aggregate diffs per line for the Source file marker
            if source_info != None:
                line = source_info.line()
                file_name = source_info.file_name()
                excess_by_line[line] += excess
                total_by_line[line] += per_instance_shared_wavefronts

    for line_number, local_diff in excess_by_line.items():
        # Create source marker in the Source file per affected line
        fe.source_marker("{:.2f}% of this line's shared wavefronts are excessive.".format(local_diff / total_by_line[line_number] * 100), line_number, NvRules.IFrontend.MarkerKind_SOURCE, file_name, NvRules.IFrontend.MsgType_MSG_WARNING)

    if total_diff > 0:
        message = "This kernel has uncoalesced shared accesses resulting in a total of {} excessive wavefronts ({:.0f}% of the total {} wavefronts)." \
            " Check the L1 Wavefronts Shared Excessive table for the primary source locations." \
            " The @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab@ has an example on optimizing shared memory accesses." \
            .format(total_diff, 100. * total_diff / total_shared_wavefronts, total_shared_wavefronts)
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message)

        speedup_type, speedup_value = get_estimated_speedup(metrics)
        fe.speedup(msg_id, speedup_type, speedup_value)

        fe.focus_metric(msg_id, metrics["excessive_wavefronts"].name(), total_diff, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Reduce the number of excessive wavefronts in L1TEX")
        fe.load_chart_from_file("UncoalescedSharedAccess.chart")
