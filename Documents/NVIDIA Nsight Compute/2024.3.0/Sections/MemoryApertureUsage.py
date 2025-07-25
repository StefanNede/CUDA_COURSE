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
import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser, Importance

requested_metrics_base = [
    MetricRequest("device__attribute_compute_capability_major", "cc_major"),
    MetricRequest("device__attribute_compute_capability_minor", "cc_minor"),
]

requested_metrics = [
    MetricRequest("lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed"),
    MetricRequest("lts__t_sectors_srcunit_tex_lookup_miss.sum"),
    MetricRequest("lts__t_sectors_srcunit_tex_aperture_peer_lookup_miss.sum"),
    MetricRequest("lts__t_sectors_srcunit_tex_aperture_sysmem_lookup_miss.sum"),
    # additional metrics for speedup estimation
    MetricRequest("dram__bytes.sum.per_second", "dram_bandwidth", Importance.OPTIONAL, 0),
    MetricRequest("pcie__read_bytes.sum.per_second", "pcie_read_bandwidth", Importance.OPTIONAL, 0),
    MetricRequest("pcie__write_bytes.sum.per_second", "pcie_write_bandwidth", Importance.OPTIONAL, 0),
    MetricRequest("nvlrx__bytes.sum.per_second", "nvlink_read_bandwidth", Importance.OPTIONAL, 0, False),
    MetricRequest("nvltx__bytes.sum.per_second", "nvlink_write_bandwidth", Importance.OPTIONAL, 0, False),
]


def get_identifier():
    return "MemoryApertureUsage"

def get_name():
    return "Memory Aperture Usage"

def get_description():
    return "Detection of frequent memory accesses backed by apertures with slower memory bandwidth and higher latency."

def get_section_identifier():
    return "MemoryWorkloadAnalysis_Chart"

def get_parent_rules_identifiers():
    return ["Memory"]

def get_estimated_speedup(metrics, aperture):
    all_lookup_misses = metrics["lts__t_sectors_srcunit_tex_lookup_miss.sum"].value()
    aperture_lookup_misses = metrics["lts__t_sectors_srcunit_tex_aperture_{}_lookup_miss.sum".format(aperture)].value()

    dram_bandwidth = metrics["dram_bandwidth"].value()
    pcie_bandwidth = metrics["pcie_read_bandwidth"].value() + metrics["pcie_write_bandwidth"].value()
    nvlink_bandwidth = metrics["nvlink_read_bandwidth"].value() + metrics["nvlink_write_bandwidth"].value()

    if aperture == "sysmem":
        # System memory is expected to be connected via PCIe
        aperture_bandwidth = pcie_bandwidth
    elif aperture == "peer":
        # Peer memory is expected to be connected via PCIe or NVLink
        aperture_bandwidth = max(pcie_bandwidth, nvlink_bandwidth)
    else:
        # unknown aperture, cannot calculate speedup
        return NvRules.IFrontend.SpeedupType_LOCAL, 0

    if all_lookup_misses != 0 and dram_bandwidth != 0 and aperture_bandwidth != 0:
        # Only give an estimate if we could collect some value for the aperture bandwidth
        improvement_percent = (aperture_lookup_misses / all_lookup_misses) * (1 - aperture_bandwidth / dram_bandwidth) * 100
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
    else:
        improvement_percent = 0
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL

    return speedup_type, improvement_percent


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics_base = RequestedMetricsParser(handle, action).parse(requested_metrics_base)

    cc = metrics_base["cc_major"].value() * 10 + metrics_base["cc_minor"].value()
    if (False
        or cc == 72
        or cc == 87
       ):
       return

    apertures = {
        "peer" : (
            "Peer"
        ),
        "sysmem" : (
            "System"
        )
    }

    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    lts__t_sectors_srcunit_tex_peak_pct = metrics["lts__t_sectors_srcunit_tex.avg.pct_of_peak_sustained_elapsed"].value()
    lts__t_sectors_srcunit_tex_lookup_miss = metrics["lts__t_sectors_srcunit_tex_lookup_miss.sum"].value()

    lts__high_utilization_threshold = 50
    lts__high_aperture_utilization_threshold = 40

    for aperture in apertures:
        aperture_info = apertures[aperture]
        metric_name = "lts__t_sectors_srcunit_tex_aperture_{}_lookup_miss.sum".format(aperture)
        lts__t_sectors_srcunit_tex_aperture_lookup_miss = metrics[metric_name].value()
        lts__t_sectors_srcunit_tex_aperture_lookup_miss_ratio = 100. * lts__t_sectors_srcunit_tex_aperture_lookup_miss / lts__t_sectors_srcunit_tex_lookup_miss if lts__t_sectors_srcunit_tex_lookup_miss else 0.

        if lts__t_sectors_srcunit_tex_peak_pct > lts__high_utilization_threshold and lts__t_sectors_srcunit_tex_aperture_lookup_miss_ratio > lts__high_aperture_utilization_threshold:
            message = "{} memory backs {:.1f}% of the data in the L2 cache that was requested by L1TEX and had cache misses in L2. ".format(aperture_info, lts__t_sectors_srcunit_tex_aperture_lookup_miss_ratio)
            message += "Fetching data from {} memory is considerably slower than accessing the device's dedicated DRAM, as the data needs to be communicated over PCIE or NVLINK. ".format(aperture_info.lower())
            message += "Consider moving frequently accessed data to DRAM before launching this kernel."
            if 80 <= cc:
                message += " Tweaking the L2 cache policies can help optimizing the cache hit rates for accesses to slower {} memory. ".format(aperture_info.lower())
                message += "Lookup CUaccessProperty and policy CU_ACCESS_PROPERTY_PERSISTING for more details."

            msg_id = fe.message(NvRules.IFrontend.MsgType_MSG_OPTIMIZATION, message, "{} Memory Usage".format(aperture_info))

            speedup_type, speedup_value = get_estimated_speedup(metrics, aperture)
            fe.speedup(msg_id, speedup_type, speedup_value)

            fe.focus_metric(msg_id, metric_name, lts__t_sectors_srcunit_tex_aperture_lookup_miss, NvRules.IFrontend.Severity_SEVERITY_DEFAULT, "Decrease the lookup misses to {} memory".format(aperture_info.lower()))
