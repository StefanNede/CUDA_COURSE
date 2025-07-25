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
    return "Occupancy"

def get_name():
    return "Occupancy"

def get_description():
    return "Occupancy section results analysis"

def get_section_identifier():
    return "Occupancy"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    theoretical_occupancy = action.metric_by_name("sm__maximum_warps_per_active_cycle_pct").as_double()
    achieved_occupancy = action.metric_by_name("sm__warps_active.avg.pct_of_peak_sustained_active").as_double()

    occupancy_difference = theoretical_occupancy - achieved_occupancy

    messages = []
    msg_type = NvRules.IFrontend.MsgType_MSG_OK

    if theoretical_occupancy == 100:
        messages.append("This kernel's theoretical occupancy is not impacted by any block limit.")
    else:
        msg_type = NvRules.IFrontend.MsgType_MSG_WARNING

        limit_types = {
            "blocks" : "the number of blocks that can fit on the SM",
            "registers" : "the number of required registers",
            "shared_mem" : "the required amount of shared memory",
            "warps" : "the number of warps within each block"
        }

        limiters = []
        for limiter in limit_types:
            limit_value = action.metric_by_name("launch__occupancy_limit_{}".format(limiter)).as_uint64()
            limit_msg = limit_types[limiter]
            limiters.append((limiter, limit_value, limit_msg))

        sorted_limiters = sorted(limiters, key=lambda limit: limit[1])
        last_limiter = -1
        for limiter in sorted_limiters:
            value = limiter[1]
            if last_limiter == -1 or value == last_limiter:
                messages.append("This kernel's theoretical occupancy ({:.1f}%) is limited by {}".format(theoretical_occupancy, limiter[2]))
                last_limiter = value

    if occupancy_difference > 10:
        msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
        messages.append("The difference between calculated theoretical ({:.1f}%) and measured achieved occupancy ({:.1f}%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution.".format(theoretical_occupancy, achieved_occupancy))
        messages.append("Load imbalances can occur between warps within a block as well as across blocks of the same kernel.")

    if len(messages) > 0:
        if msg_type == NvRules.IFrontend.MsgType_MSG_WARNING:
            messages.append("See the @url:CUDA Best Practices Guide:https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy@ for more details on optimizing occupancy.")
        fe.message(msg_type, " ".join(messages), "Occupancy Limiters")
