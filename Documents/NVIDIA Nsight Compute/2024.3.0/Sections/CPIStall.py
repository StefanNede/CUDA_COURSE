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
import re

import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser, Importance

requested_metrics = [
    MetricRequest("smsp__issue_active.avg.per_cycle_active", "issue_active"),
    MetricRequest("smsp__average_warp_latency_per_inst_issued.ratio", "warp_cycles_per_issue"),
    # metrics for each stall reason
    MetricRequest("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", "smsp_average_barrier", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_branch_resolving_per_issue_active.ratio", "smsp_average_branch_resolving", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active.ratio", "smsp_average_dispatch_stall", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_drain_per_issue_active.ratio", "smsp_average_drain", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio", "smsp_average_warpgroup_arrive", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__average_warps_issue_stalled_imc_miss_per_issue_active.ratio", "smsp_average_imc_miss", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", "smsp_average_lg_throttle", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "smsp_average_long_scoreboard", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "smsp_average_math_pipe_throttle", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_membar_per_issue_active.ratio", "smsp_average_membar", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "smsp_average_mio_throttle", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__average_warps_issue_stalled_misc_per_issue_active.ratio", "smsp_average_misc", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio", "smsp_average_no_instruction", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", "smsp_average_not_selected", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio", "smsp_average_short_scoreboard", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_sleeping_per_issue_active.ratio", "smsp_average_sleeping", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.ratio", "smsp_average_tex_throttle", Importance.OPTIONAL, None),
    MetricRequest("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", "smsp_average_wait", Importance.OPTIONAL, None),
    # pc sampling metrics used for source markers (collected in SourceCounters.section)
    MetricRequest("smsp__pcsamp_sample_count", "pc_sampling_count", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_barrier", "pc_sampling_barrier", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_branch_resolving", "pc_sampling_branch_resolving", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_dispatch_stall", "pc_sampling_dispatch_stall", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_drain", "pc_sampling_drain", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_imc_miss", "pc_sampling_imc_miss", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_lg_throttle", "pc_sampling_lg_throttle", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_long_scoreboard", "pc_sampling_long_scoreboard", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_math_pipe_throttle", "pc_sampling_math_pipe_throttle", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_membar", "pc_sampling_membar", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_mio_throttle", "pc_sampling_mio_throttle", Importance.OPTIONAL, None, False),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_misc", "pc_sampling_misc", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_no_instructions", "pc_sampling_no_instruction", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_not_selected", "pc_sampling_not_selected", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_short_scoreboard", "pc_sampling_short_scoreboard", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_sleeping", "pc_sampling_sleeping", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_tex_throttle", "pc_sampling_tex_throttle", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_wait", "pc_sampling_wait", Importance.OPTIONAL, None),
    MetricRequest("smsp__pcsamp_warps_issue_stalled_warpgroup_arrive", "pc_sampling_warpgroup_arrive", Importance.OPTIONAL, None, False),
]


def get_identifier():
    return "CPIStall"

def get_name():
    return "Warp Stall"

def get_description():
    return "Warp stall analysis"

def get_section_identifier():
    return "WarpStateStats"

def get_parent_rules_identifiers():
    return ["IssueSlotUtilization"]

def get_estimated_speedup(parent_weights, warp_cycles_per_stall, warp_cycles_per_issue):
    improvement_local = warp_cycles_per_stall / warp_cycles_per_issue

    parent_speedup_name = "issue_slot_util_speedup_normalized"
    if parent_speedup_name in parent_weights:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
        improvement_global = min(parent_weights[parent_speedup_name], improvement_local)
        improvement_percent = improvement_global * 100
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL
        improvement_percent = improvement_local * 100

    return speedup_type, improvement_percent

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)
    parent_weights = fe.receive_dict_from_parent("IssueSlotUtilization")

    # placeholders are substituted by apply_kb.py with content from kb.json during build time
    stall_types = {
        "barrier" : (
            "Warp was stalled waiting for sibling warps at a CTA barrier. A high number of warps waiting at a barrier is commonly caused by diverging code paths before a barrier. This causes some warps to wait a long time until other warps reach the synchronization point. Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512 threads or greater, consider splitting it into smaller groups. This can increase eligible warps without affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which barrier instruction causes the most stalls, and optimize the code executed before that synchronization point first.",
            None),
        "branch_resolving" : (
            "Warp was stalled waiting for a branch target to be computed, and the warp program counter to be updated. To reduce the number of stalled cycles, consider using fewer jump/branch operations and reduce control flow divergence, e.g. by reducing or coalescing conditionals in your code. See also the related No Instructions state.",
            None),
        "dispatch_stall" : (
            "Warp was stalled waiting on a dispatch stall. A warp stalled during dispatch has an instruction ready to issue, but the dispatcher holds back issuing the warp due to other conflicts or events.",
            None),
        "drain" : (
            "Warp was stalled after EXIT waiting for all outstanding memory operations to complete so that warp's resources can be freed. A high number of stalls due to draining warps typically occurs when a lot of data is written to memory towards the end of a kernel. Make sure the memory access patterns of these store operations are optimal for the target architecture and consider parallelized data reduction, if applicable.",
            None),
        "imc_miss" : (
            "Warp was stalled waiting for an immediate constant cache (IMC) miss. A read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache. Immediate constants are encoded into the SASS instruction as 'c[bank][offset]'. Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp. As such, the constant cache is best when threads in the same warp access only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access.",
            None),
        "lg_throttle" : (
            "Warp was stalled waiting for the L1 instruction queue for local and global (LG) memory operations to be not full. Typically, this stall occurs only when executing local or global memory instructions extremely frequently. Avoid redundant global memory accesses. Try to avoid using thread-local memory by checking if dynamically indexed arrays are declared in local scope, of if the kernel has excessive register pressure causing by spills. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions.",
            None),
        "long_scoreboard" : (
            "Warp was stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory.",
            None),
        "math_pipe_throttle" : (
            "Warp was stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way.",
            None),
        "membar" : (
            "Warp was stalled waiting on a memory barrier. Avoid executing any unnecessary memory barriers and assure that any outstanding memory operations are fully optimized for the target architecture.",
            None),
        "mio_throttle" : (
            "Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure.",
            None),
        "misc" : (
            "Warp was stalled for a miscellaneous hardware reason.",
            None),
        "no_instruction" : (
            "Warp was stalled waiting to be selected to fetch an instruction or waiting on an instruction cache miss. A high number of warps not having an instruction fetched is typical for very short kernels with less than one full wave of work in the grid. Excessively jumping across large blocks of assembly code can also lead to more warps stalled for this reason, if this causes misses in the instruction cache. See also the related Branch Resolving state.",
            None),
        "not_selected" : (
            "Warp was stalled waiting for the micro scheduler to select the warp to issue. Not selected warps are eligible warps that were not picked by the scheduler to issue that cycle as another warp was selected. A high number of not selected warps typically means you have sufficient warps to cover warp latencies and you may consider reducing the number of active warps to possibly increase cache coherence and data locality.",
            None),
        "short_scoreboard" : (
            "Warp was stalled waiting for a scoreboard dependency on a MIO (memory input/output) operation (not to L1TEX). The primary reason for a high number of stalls due to short scoreboards is typically memory operations to shared memory. Other reasons include frequent execution of special math instructions (e.g. MUFU) or dynamic branching (e.g. BRX, JMX). Consult the Memory Workload Analysis section to verify if there are shared memory operations and reduce bank conflicts, if reported. Assigning frequently accessed values to variables can assist the compiler in using low-latency registers instead of direct memory accesses.",
            None),
        "sleeping" : (
            "Warp was stalled due to all threads in the warp being in the blocked, yielded, or sleep state. Reduce the number of executed NANOSLEEP instructions, lower the specified time delay, and attempt to group threads in a way that multiple threads in a warp sleep at the same time.",
            None),
        "tex_throttle" : (
            "Warp was stalled waiting for the L1 instruction queue for texture operations to be not full. This stall reason is high in cases of extreme utilization of the L1TEX pipeline. Try issuing fewer texture fetches, surface loads, surface stores, or decoupled math operations. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions. Consider converting texture lookups or surface loads into global memory lookups. Texture can accept four threads' requests per cycle, whereas global accepts 32 threads.",
            None),
        "wait" : (
            "Warp was stalled waiting on a fixed latency execution dependency. Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options.",
            None),
        "warpgroup_arrive" : (
            "Warp was stalled waiting on a WARPGROUP.ARRIVES or WARPGROUP.WAIT instruction.",
            None),
    }

    issue_active = metrics["issue_active"].value()
    warp_cycles_per_issue = metrics["warp_cycles_per_issue"].value()

    total_sample_count_ratio = 0.1
    high_stall_ratio = 0.3
    sample_count_metric = metrics["pc_sampling_count"]

    for stall_name in stall_types:
        stall_metric_name = f"pc_sampling_{stall_name}"
        if metrics[stall_metric_name] is not None:
            stall_metric = metrics[stall_metric_name]
            correlation_ids = stall_metric.correlation_ids()

            if stall_metric.num_instances() == sample_count_metric.num_instances():
                for i in range(stall_metric.num_instances()):
                    # True if this particular stall metric is responsible for at least high_stall_ratio of sampled stalls in this instance
                    percent_of_local_stalls_condition = sample_count_metric.as_uint64(i) != 0 and stall_metric.as_uint64(i) / sample_count_metric.as_uint64(i) > high_stall_ratio
                    # True if this instance is responsible for at least total_sample_count_ratio of all total sampled stalls
                    percent_of_total_stalls_condition = sample_count_metric.as_uint64() != 0 and sample_count_metric.as_uint64(i) / sample_count_metric.as_uint64() > total_sample_count_ratio
                    if percent_of_local_stalls_condition and percent_of_total_stalls_condition:
                        address = correlation_ids.as_uint64(i)
                        fe.source_marker("This line is responsible for {:.1f}% of all warp stalls. {:.1f}% of the stalls for this line are of type {}.".format(sample_count_metric.as_uint64(i) / sample_count_metric.as_uint64() * 100, stall_metric.as_uint64(i) / sample_count_metric.as_uint64(i) * 100, stall_name), address, NvRules.IFrontend.MarkerKind_SASS, NvRules.IFrontend.MsgType_MSG_WARNING)
                        source_info = action.source_info(address)
                        if source_info != None:
                            line_number = source_info.line()
                            file_name = source_info.file_name()
                            fe.source_marker("This line is responsible for a high number of warp stalls. See markers on SASS lines for details.", line_number, NvRules.IFrontend.MarkerKind_SOURCE, file_name, NvRules.IFrontend.MsgType_MSG_WARNING)

    reported_stalls = []
    for stall_name in stall_types:
        stall_metric_name = f"smsp_average_{stall_name}"
        if metrics[stall_metric_name] is None:
            continue

        warp_cycles_per_stall = metrics[stall_metric_name].value()

        issue_active_threshold = 0.8
        ratio_threshold = 0.3
        if issue_active < issue_active_threshold and warp_cycles_per_issue > 0 and ratio_threshold < (warp_cycles_per_stall / warp_cycles_per_issue):
            warp_cycles_avg = 100. * warp_cycles_per_stall / warp_cycles_per_issue
            stall_info = stall_types[stall_name]
            stall_description = re.sub("^Warp was stalled ", "", stall_info[0])
            stall_extra = stall_info[1]
            message = "On average, each warp of this kernel spends {:.1f} cycles being stalled {}".format(warp_cycles_per_stall, stall_description)
            message += " This stall type represents about {:.1f}% of the total average of {:.1f} cycles between issuing two instructions.".format(warp_cycles_avg, warp_cycles_per_issue)
            if stall_extra:
                message += " " + stall_extra

            speedup_type, speedup_value = get_estimated_speedup(parent_weights, warp_cycles_per_stall, warp_cycles_per_issue)
            focus_metrics = [
                (metrics["issue_active"].name(), issue_active, NvRules.IFrontend.Severity_SEVERITY_HIGH, "Increase the average number of instructions issued per cycle"),
                (stall_metric_name, warp_cycles_per_stall, NvRules.IFrontend.Severity_SEVERITY_DEFAULT if warp_cycles_per_issue > 10 else NvRules.IFrontend.Severity_SEVERITY_LOW, "Decrease the number of cycles spent in {} stalls".format(stall_name.replace("_", " ")))]
            reported_stalls.append((stall_name, warp_cycles_per_stall, message, focus_metrics, speedup_type, speedup_value))

    sorted_stalls = sorted(reported_stalls, key=lambda stall: stall[1], reverse=True)
    for stall in sorted_stalls:
        message_name = stall[0].replace("_", " ").title() + " Stalls"
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, stall[2], message_name)
        speedup_type, speedup_value = stall[4], stall[5]
        fe.speedup(msg_id, speedup_type, speedup_value)
        for fm in stall[3]:
            fe.focus_metric(msg_id, fm[0], fm[1], fm[2], fm[3])

    if len(sorted_stalls) > 0:
        fe.message(NvRules.IFrontend.MsgType_MSG_OK, \
            'Check the @section:SourceCounters:Warp Stall Sampling (All Samples)@ table for the top stall locations in your source based on sampling data.'\
            ' The @url:Kernel Profiling Guide:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference@ provides more details on each stall reason.')
