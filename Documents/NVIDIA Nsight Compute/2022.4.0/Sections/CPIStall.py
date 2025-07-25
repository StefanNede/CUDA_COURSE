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
    return "CPIStall"

def get_name():
    return "Warp Stall"

def get_description():
    return "Warp stall analysis"

def get_section_identifier():
    return "WarpStateStats"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    stall_types = {
        "barrier" : (
            "waiting for sibling warps at a CTA barrier",
            "A high number of warps waiting at a barrier is commonly caused by diverging code paths before a barrier that causes some warps to wait a long time until other warps reach the synchronization point. Whenever possible try to divide up the work into blocks of uniform workloads. Use the Source View's sampling columns to identify which barrier instruction causes the most stalls and optimize the code executed before that synchronization point first."),
        "branch_resolving" : (
            "waiting for a branch target to be computed, and the warp program counter to be updated",
            "To reduce the number of stalled cycles, consider using fewer jump/branch operations and reduce control flow divergence, e.g. by reducing or coalescing conditionals in your code."),
        "dispatch_stall" : (
            "waiting on a dispatch stall",
            "A warp stalled during dispatch has a ready instruction, but the dispatcher holds back issuing the warp due to other conflicts or events."),
        "drain" : (
            "waiting after an EXIT instruction for all outstanding memory instructions to complete so that the warp's resources can be freed",
            "A high number of stalls due to draining warps typically occurs when a lot of data is written to memory towards the end of a kernel. Make sure the memory access patterns of these store operations are optimal for the target architecture and consider parallelized data reduction, if applicable."),
        "imc_miss" : (
            "waiting for an immediate constant cache (IMC) miss",
            "A read from constant memory costs one memory read from global memory only on a cache miss; otherwise, it just costs one read from the constant cache. Immediate constants are encoded into the SASS instruction as 'c[bank][offset]'. All threads access the same value. Consider computing values instead of loading them as constant, when applicable."),
        "lg_throttle" : (
            "waiting for the local/global instruction queue to be not full",
            "Typically this stall occurs only when executing local or global memory instructions extremely frequently. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions."),
        "long_scoreboard" : (
            "waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture, rtcore) operation",
            "To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality or by changing the cache configuration, and consider moving frequently used data to registers and to shared memory."),
        "math_pipe_throttle" : (
            "waiting for a math execution pipeline to be available",
            "This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way."),
        "membar" : (
            "waiting on a memory barrier",
            "Avoid executing any unnecessary memory barriers and assure that any outstanding memory operations are fully optimized for the target architecture."),
        "mio_throttle" : (
            "waiting for an MIO instruction queue to be not full",
            "This stall reason is high in cases of utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline pressure."),
        "misc" : (
            "on a miscellaneous hardware reason",
            None),
        "no_instruction" : (
            "due to not having the next instruction fetched yet",
            "A high number of warps not having an instruction fetched is typical for very short kernels with less than one full wave of work in the grid. Excessively jumping across large blocks of assembly code can also lead to more warps stalled for this reason, if this causes misses in the instruction cache."),
        "not_selected" : (
            "due to not being selected by the scheduler",
            "Not selected warps are eligible warps that were not picked by the scheduler to issue that cycle as another warp was selected. A high number of not selected warps typically means you have sufficient warps to cover warp latencies and you may consider reducing the number of active warps to possibly increase cache coherence and data locality."),
        "short_scoreboard" : (
            "waiting for a scoreboard dependency on an MIO operation (not to TEX or L1)",
            "The primary reason for a high number of stalls due to short scoreboards is typically memory operations to shared memory, but other contributors include frequent execution of special math instructions (e.g. MUFU) or dynamic branching (e.g. BRX, JMX). Consult the Memory Workload Analysis section to verify if there are shared memory operations and reduce bank conflicts, if reported. Assigning frequently accessed values to variables can assist the compiler in using low-latency registers instead of direct memory accesses."),
        "sleeping" : (
            "waiting for a thread in the warp to come out of the sleep state",
            "Reduce the number of executed NANOSLEEP instructions, lower the specified time delay, and attempt to group threads in a way that multiple threads in a warp sleep at the same time."),
        "tex_throttle" : (
            "waiting for the L1TEX instruction queue to be not full",
            "This stall reason is high in cases of utilization of the L1TEX pipeline. If applicable, consider combining multiple lower-width memory operations into fewer wider memory operations and try interleaving memory operations and math instructions."),
        "wait" : (
            "on a fixed latency execution dependency",
            "Typically, this stall reason should be very low and only shows up as a top contributor in already highly optimized kernels. Try to hide the corresponding instruction latencies by increasing the number of active warps, restructuring the code or unrolling loops. Furthermore, consider switching to lower-latency instructions, e.g. by making use of fast math compiler options."),
    }

    report_metrics = action.metric_names()

    issue_active = action.metric_by_name("smsp__issue_active.avg.per_cycle_active").as_double()
    warp_cycles_per_issue = action.metric_by_name("smsp__average_warp_latency_per_inst_issued.ratio").as_double()

    reported_stalls = []
    for stall_name in stall_types:
        stall_metric_name = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio".format(stall_name)
        if not stall_metric_name in report_metrics:
            continue

        warp_cycles_per_stall = action.metric_by_name(stall_metric_name).as_double()

        issue_active_threshold = 0.8
        ratio_threshold = 0.3
        if issue_active < issue_active_threshold and warp_cycles_per_issue > 0 and ratio_threshold < (warp_cycles_per_stall / warp_cycles_per_issue):
            warp_cycles_avg = 100. * warp_cycles_per_stall / warp_cycles_per_issue
            stall_info = stall_types[stall_name]
            reason_short = stall_info[0]
            reason_detailed = stall_info[1]
            message = "On average, each warp of this kernel spends {:.1f} cycles being stalled {}. This represents about {:.1f}% of the total average of {:.1f} cycles between issuing two instructions.".format(warp_cycles_per_stall, reason_short, warp_cycles_avg, warp_cycles_per_issue)
            if reason_detailed:
                message += " " + reason_detailed

            focus_metrics = [
                ("smsp__issue_active.avg.per_cycle_active", issue_active, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{:.3f} < {:.1f}".format(issue_active, issue_active_threshold)),
                (stall_metric_name, warp_cycles_per_stall, NvRules.IFrontend.Severity_SEVERITY_HIGH if warp_cycles_per_issue > 10 else NvRules.IFrontend.Severity_SEVERITY_LOW, "{:.2f} / {:.2f} > {:.1f}".format(warp_cycles_per_stall, warp_cycles_per_issue, ratio_threshold))]
            reported_stalls.append((stall_name, warp_cycles_per_stall, message, focus_metrics))

    sorted_stalls = sorted(reported_stalls, key=lambda stall: stall[1], reverse=True)
    for stall in sorted_stalls:
        msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, stall[2], stall[0])
        for fm in stall[3]:
            fe.focus_metric(msg_id, fm[0], fm[1], fm[2], fm[3])


    if len(sorted_stalls) > 0:
        fe.message(NvRules.IFrontend.MsgType_MSG_OK, \
            'Check the @section:SourceCounters:Source Counters@ section for the top stall locations in your source based on sampling data.'\
            ' The @url:Kernel Profiling Guide:https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sampling@ provides more details on each stall reason.')

