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
    return "MemoryL2Compression"

def get_name():
    return "Memory L2 Compression"

def get_description():
    return "Detection of inefficient use of L2 Compute Data Compression"

def get_section_identifier():
    return "MemoryWorkloadAnalysis_Chart"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    cc_major = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    cc_minor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    cc = cc_major * 10 + cc_minor

    low_compression_threshold = 20

    if 90 <= cc:
        return

    if 80 <= cc:
        compInputSectors = action.metric_by_name("lts__gcomp_input_sectors.sum").as_double()
        if 0.0 < compInputSectors:
            compInputBytes = 32.0 * compInputSectors
            compSuccessRate = action.metric_by_name("lts__average_gcomp_input_sector_success_rate.pct").as_double()

            # Check for low success rate
            if compSuccessRate < low_compression_threshold:
                message = "Out of the {:.1f} bytes sent to the L2 Compression unit only {:.2f}% were successfully compressed. To increase this success rate, consider marking only those memory regions as compressible that contain the most zero values and/or expose the most homogeneous values.".format(compInputBytes, compSuccessRate)
                fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message, "Low Compression Rate")

            l1WriteBytes = action.metric_by_name("l1tex__m_l1tex2xbar_write_bytes.sum").as_double()
            # Check for write access pattern
            if (l1WriteBytes > 0) and (4.0 * l1WriteBytes < compInputBytes):
                message = "The access patterns for writes to compressible memory are not well suited for the L2 Compression unit. As a consequence, {:.1f}x the data written to the L2 cache has to be communicated to the L2 Compression unit. Try maximizing local coherence for the write operations to compressible memory. For example, avoid writes with large strides as they lead to partial accesses to many L2 cache lines. Instead, try accessing fewer overall cache lines by modifying many values per cache line with each warp's execution of a write operation.".format(compInputBytes/l1WriteBytes)
                fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message, "Access Pattern")