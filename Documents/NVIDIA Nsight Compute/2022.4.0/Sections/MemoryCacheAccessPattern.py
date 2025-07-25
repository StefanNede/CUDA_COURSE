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
    return "MemoryCacheAccessPattern"

def get_name():
    return "Memory Cache Access Pattern"

def get_description():
    return "Detection of inefficient memory access patterns in the L1TEX cache and L2 cache."

def get_section_identifier():
    return "MemoryWorkloadAnalysis_Tables"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    # Metrics ==========================================================================================================
    smsp__inst_executed_op_memory_8b = action.metric_by_name("smsp__sass_inst_executed_op_memory_8b.sum").as_double()
    smsp__inst_executed_op_memory_16b = action.metric_by_name("smsp__sass_inst_executed_op_memory_16b.sum").as_double()
    smsp__inst_executed_op_memory_32b = action.metric_by_name("smsp__sass_inst_executed_op_memory_32b.sum").as_double()
    smsp__inst_executed_op_memory_64b = action.metric_by_name("smsp__sass_inst_executed_op_memory_64b.sum").as_double()
    smsp__inst_executed_op_memory_128b = action.metric_by_name("smsp__sass_inst_executed_op_memory_128b.sum").as_double()
    cc_major = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    cc_minor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    cc = cc_major * 10 + cc_minor

    # Derived Metrics --------------------------------------------------------------------------------------------------
    smsp__inst_executed_op_memory_flat_sum = \
        smsp__inst_executed_op_memory_8b \
        + smsp__inst_executed_op_memory_16b \
        + smsp__inst_executed_op_memory_32b \
        + smsp__inst_executed_op_memory_64b \
        + smsp__inst_executed_op_memory_128b

    smsp__inst_executed_op_memory_weighted_sum = \
        8 * smsp__inst_executed_op_memory_8b \
        + 16 * smsp__inst_executed_op_memory_16b \
        + 32 * smsp__inst_executed_op_memory_32b \
        + 64 * smsp__inst_executed_op_memory_64b \
        + 128 * smsp__inst_executed_op_memory_128b

    smspAvgMemoryBytesPerInst = smsp__inst_executed_op_memory_weighted_sum / smsp__inst_executed_op_memory_flat_sum / 8 if smsp__inst_executed_op_memory_flat_sum > 0 else 0

    # L1TEX ============================================================================================================
    l1tex_access_types = {
        "mem_global_op_ld" : (
            "Global Load"),
        "mem_global_op_st" : (
            "Global Store"),
        "mem_local_op_ld" : (
            "Local Load"),
        "mem_local_op_st" : (
            "Local Store"),
    }

    for access_type in l1tex_access_types:
        access_info = l1tex_access_types[access_type]
        sectors = action.metric_by_name("l1tex__t_sectors_pipe_lsu_{}.sum".format(access_type)).as_double()
        requests = action.metric_by_name("l1tex__t_requests_pipe_lsu_{}.sum".format(access_type)).as_double()
        sectors_per_request = sectors / requests if requests > 0 else 0

        if sectors > 0 and requests > 0 and sectors_per_request > smspAvgMemoryBytesPerInst:
            message = "The memory access pattern for {}s in L1TEX might not be optimal. ".format(access_info.lower())
            message += "On average, this kernel accesses {:.1f} bytes per thread per memory request; ".format(smspAvgMemoryBytesPerInst)
            message += "but the address pattern, possibly caused by the stride between threads, results in {:.1f} sectors per request, or {:.1f}*32 = {:.1f} bytes of cache data transfers per request. ".format(sectors_per_request,sectors_per_request,32 * sectors_per_request)
            message += "The optimal thread address pattern for {:.1f} byte accesses would result in {:.1f}*32 = {:.1f} bytes of cache data transfers per request, to maximize L1TEX cache performance. ".format(smspAvgMemoryBytesPerInst,smspAvgMemoryBytesPerInst,32 * smspAvgMemoryBytesPerInst)
            message += "Check the @section:SourceCounters:Source Counters@ section for uncoalesced {}s.".format(access_info.lower())
            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message, "L1TEX {} Access Pattern".format(access_info))
            fe.focus_metric(msg_id, "Sectors per L1TEX Request", sectors_per_request, NvRules.IFrontend.Severity_SEVERITY_HIGH if sectors_per_request > 2 * smspAvgMemoryBytesPerInst else NvRules.IFrontend.Severity_SEVERITY_LOW, "{:,.0f} / {:,.0f} > {:.1f}".format(sectors, requests, smspAvgMemoryBytesPerInst))

    # L2 ==============================================================================================================
    l2_access_types = {
        "tex_op_read" : (
            "Load"),
        "tex_op_write" : (
            "Store"),
    }

    for access_type in l2_access_types:
        access_info = l2_access_types[access_type]
        sectors = action.metric_by_name("lts__t_sectors_srcunit_{}.sum".format(access_type)).as_double()
        requests = action.metric_by_name("lts__t_requests_srcunit_{}.sum".format(access_type)).as_double()
        sectors_per_request = sectors / requests if requests > 0 else 0

        # Anything less than 4 is not ideal, but we don't want to show a warning if it's very close.
        if sectors > 0 and requests > 0 and sectors_per_request < 3.5:
            message = "The memory access pattern for {}s from L1TEX to L2 is not optimal. ".format(access_info.lower())
            message += "The granularity of an L1TEX request to L2 is a 128 byte cache line. That is 4 consecutive 32-byte sectors per L2 request. "
            message += "However, this kernel only accesses an average of {:.1f} sectors out of the possible 4 sectors per cache line. ".format(sectors_per_request)
            message += "Check the @section:SourceCounters:Source Counters@ section for uncoalesced {}s and try to minimize how many cache lines need to be accessed per memory request.".format(access_info.lower())
            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message, "L2 {} Access Pattern".format(access_info))
            fe.focus_metric(msg_id, "Sectors per L2 Request", sectors_per_request, NvRules.IFrontend.Severity_SEVERITY_HIGH if sectors_per_request <= 2 else NvRules.IFrontend.Severity_SEVERITY_LOW, "{:,.0f} / {:,.0f} < 4".format(sectors, requests))

    # DRAM ============================================================================================================
    if (True
        and cc != 72
        and cc != 87
       ):
        dram__read_peak_pct = action.metric_by_name("dram__bytes_read.sum.pct_of_peak_sustained_elapsed").as_double()
        lts__read_sectors = action.metric_by_name("lts__t_sectors_srcunit_tex_op_read.sum").as_double()
        lts__read_sectors_hits = action.metric_by_name("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum").as_double()
        lts__read_sectors_misses = action.metric_by_name("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum").as_double()
        lts__read_sectors_not_hit = lts__read_sectors - lts__read_sectors_hits

        if dram__read_peak_pct > 50 and lts__read_sectors_not_hit < lts__read_sectors_misses:
            message = "The memory access pattern for loads from device memory causes {:,.0f} sectors to be read from DRAM, which is {:.1f}x of the {:,.0f} sectors causing a miss in the L2 cache. ".format(lts__read_sectors_misses, lts__read_sectors_misses/lts__read_sectors_not_hit, lts__read_sectors_not_hit)
            message += "The DRAM fetch granularity for read misses in L2 is 64 bytes, i.e. the lower or upper half of an L2 cache line. "
            message += "Try changing your access pattern to make use of both sectors returned by a DRAM read request for optimal usage of the DRAM throughput. "
            message += "For strided memory reads, avoid strides of 64 bytes or larger to avoid moving unused sectors from DRAM to L2. "
            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message, "DRAM Excessive Read Sectors")
            fe.focus_metric(msg_id, "DRAM Read Peak Utilization", dram__read_peak_pct, NvRules.IFrontend.Severity_SEVERITY_HIGH if dram__read_peak_pct > 75 else NvRules.IFrontend.Severity_SEVERITY_LOW, "{:.1f}% > 50%".format(dram__read_peak_pct))
            fe.focus_metric(msg_id, "DRAM Excessive Read Sectors", lts__read_sectors_misses - lts__read_sectors, NvRules.IFrontend.Severity_SEVERITY_HIGH, "{:,.0f} > {:,.0f}".format(lts__read_sectors_misses, lts__read_sectors))
