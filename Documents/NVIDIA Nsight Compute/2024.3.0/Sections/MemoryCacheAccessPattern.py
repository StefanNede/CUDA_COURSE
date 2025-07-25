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
from collections import namedtuple
from itertools import product

import NvRules
from RequestedMetrics import MetricRequest, RequestedMetricsParser, Importance

requested_metrics = [
    # SASS metrics bytes per sector for (global/local) x (load/store) memory accesses
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio", "bytes_per_sector_global_load"),
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.max_rate", "max_bytes_per_sector_global_load"),
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio", "bytes_per_sector_global_store"),
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.max_rate", "max_bytes_per_sector_global_store"),
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_local_op_ld.ratio", "bytes_per_sector_local_load"),
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_local_op_ld.max_rate", "max_bytes_per_sector_local_load"),
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_local_op_st.ratio", "bytes_per_sector_local_store"),
    MetricRequest("smsp__sass_average_data_bytes_per_sector_mem_local_op_st.max_rate", "max_bytes_per_sector_local_store"),
    # L1TEX/L2 global/local, load/store hit rates
    MetricRequest(
        "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
        "l1tex_global_load_hit_rate_percent",
    ),
    MetricRequest(
        "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct",
        "l1tex_global_store_hit_rate_percent",
    ),
    MetricRequest(
        "l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct",
        "l1tex_local_load_hit_rate_percent",
    ),
    MetricRequest(
        "l1tex__t_sector_pipe_lsu_mem_local_op_st_hit_rate.pct",
        "l1tex_local_store_hit_rate_percent",
    ),
    MetricRequest(
        "lts__t_sector_op_read_hit_rate.pct",
        "l2_load_hit_rate_percent",
    ),
    MetricRequest(
        "lts__t_sector_op_write_hit_rate.pct",
        "l2_store_hit_rate_percent",
    ),
    # L1TEX/L2/DRAM throughput metrics
    MetricRequest(
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex_throughput_percent",
    ),
    MetricRequest(
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2_throughput_percent",
    ),
    MetricRequest(
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram_throughput_percent",
        Importance.OPTIONAL,
        None,
        False,
    ),
]

FocusMetric = namedtuple(
    "FocusMetric",
    ["name", "value", "severity", "advice"],
)

RuleResult = namedtuple(
    "RuleResult",
    ["message", "title", "speedup_type", "speedup_value", "focus_metrics"],
    defaults=("", "", NvRules.IFrontend.SpeedupType_LOCAL, 0, []),
)


def get_identifier():
    return "MemoryCacheAccessPattern"


def get_name():
    return "Memory Cache Access Pattern"


def get_description():
    return "Detection of inefficient memory access patterns in the L1/TEX and L2 caches."


def get_section_identifier():
    return "MemoryWorkloadAnalysis_Tables"


def get_parent_rules_identifiers():
    return ["Memory"]


def get_bytes_per_sector_metric_names(memory_space, operation):
    metric_extension = f"{memory_space}_{operation}"
    bytes_per_sector_name = f"bytes_per_sector_{metric_extension}"
    max_bytes_per_sector_name = f"max_bytes_per_sector_{metric_extension}"
    return bytes_per_sector_name, max_bytes_per_sector_name


def get_bytes_per_sector_metrics(metrics, memory_space, operation):
    bytes_per_sector_name, max_bytes_per_sector_name = \
        get_bytes_per_sector_metric_names(memory_space, operation)
    bytes_per_sector = metrics[bytes_per_sector_name].value()
    max_bytes_per_sector = metrics[max_bytes_per_sector_name].value()
    return bytes_per_sector, max_bytes_per_sector


def get_speedup_and_focus_metrics(cache, memory_space, operation, metrics):
    """Speedup Estimation and Focus Metrics for memory access patterns in all caches.

    Assuming that at each cache level the bandwidth is independent of the amount
    of data moved, the speedup can be estimated as follows:

      s = time_old / time_new
        = (data_old * bandwidth_new) / (data_new * bandwidth_old)
        = (data_old / data_new)
        = (max_bytes_per_sector_old * num_sectors_old)
              / (max_bytes_per_sector_new * num_sectors_new)
        = (num_sectors_new * bytes_per_sector_new / bytes_per_sector_old)
              / num_sectors_new
        = bytes_per_sector_new / bytes_per_sector_old

    where we used that the "useful" amount of data moved, remains constant, i.e.
    bytes_per_sector_old * num_sectors_old = bytes_per_sector_new * num_sectors_new.
    Thus, the maximal speedup is s_max = max_bytes_per_sector / bytes_per_sector.
    Using that the maximal improvement = 1 - (1 / s_max), we get

      improvement_percent = (1 - bytes_per_sector / max_bytes_per_sector) * 100

    At each cache level, the relevant amount of data is given by the sectors missed
    at its respective lower-level cache, introducing new factors of `cache_miss_rate`.
    To get a global estimate, we can use the cache's throughput as a weight.
    """
    bytes_per_sector, max_bytes_per_sector = \
        get_bytes_per_sector_metrics(metrics, memory_space, operation)

    # Only get an estimate for non-trivial amounts of transferred data
    if bytes_per_sector == 0 or max_bytes_per_sector == 0:
        return NvRules.IFrontend.SpeedupType_LOCAL, 0

    # Get L1TEX miss rate for L2/DRAM, and L2 miss rate for DRAM
    l1_miss_rate = None
    l2_miss_rate = None
    l1_hit_rate_name = None
    l2_hit_rate_name = None
    if cache in ["l2", "dram"]:
        if memory_space == "global" and operation == "load":
            l1_miss_rate = 1 - metrics["l1tex_global_load_hit_rate_percent"].value() / 100
            l1_hit_rate_name = metrics["l1tex_global_load_hit_rate_percent"].name()
        elif memory_space == "global" and operation == "store":
            l1_miss_rate = 1 - metrics["l1tex_global_store_hit_rate_percent"].value() / 100
            l1_hit_rate_name = metrics["l1tex_global_store_hit_rate_percent"].name()
        elif memory_space == "local" and operation == "load":
            l1_miss_rate = 1 - metrics["l1tex_local_load_hit_rate_percent"].value() / 100
            l1_hit_rate_name = metrics["l1tex_local_load_hit_rate_percent"].name()
        elif memory_space == "local" and operation == "store":
            l1_miss_rate = 1 - metrics["l1tex_local_store_hit_rate_percent"].value() / 100
            l1_hit_rate_name = metrics["l1tex_local_store_hit_rate_percent"].name()
    if cache == "dram":
        if operation == "load":
            l2_miss_rate = 1 - metrics["l2_load_hit_rate_percent"].value() / 100
            l2_hit_rate_name = metrics["l2_load_hit_rate_percent"].name()
        elif operation == "store":
            l2_miss_rate = 1 - metrics["l2_store_hit_rate_percent"].value() / 100
            l2_hit_rate_name = metrics["l2_store_hit_rate_percent"].name()

    # Get throughput of current cache level to use as weight in global estimates
    throughput = None
    throughput_name = None
    if cache == "l1tex":
        throughput = metrics["l1tex_throughput_percent"].value() / 100
        throughput_name = metrics["l1tex_throughput_percent"].name()
    elif cache == "l2":
        throughput = metrics["l2_throughput_percent"].value() / 100
        throughput_name = metrics["l2_throughput_percent"].name()
    elif (cache == "dram") and (metrics["dram_throughput_percent"] is not None):
        throughput = metrics["dram_throughput_percent"].value() / 100
        throughput_name = metrics["dram_throughput_percent"].name()

    if throughput:
        speedup_type = NvRules.IFrontend.SpeedupType_GLOBAL
    else:
        speedup_type = NvRules.IFrontend.SpeedupType_LOCAL

    # Calculate speedup as described above
    improvement_percent = (
        (1 - bytes_per_sector / max_bytes_per_sector)
        * (l1_miss_rate if l1_miss_rate else 1)
        * (l2_miss_rate if l2_miss_rate else 1)
        * (throughput if throughput else 1)
        * 100
    )

    # Store Focus Metrics for all metrics that enter the speedup calculation
    focus_metrics = []
    bytes_per_sector_focus_metric = FocusMetric(
        metrics[get_bytes_per_sector_metric_names(memory_space, operation)[0]].name(),
        bytes_per_sector,
        NvRules.IFrontend.Severity_SEVERITY_HIGH,
        f"Increase the average number of bytes utilized per sector towards "
        f"{max_bytes_per_sector:.0f} bytes"
    )
    focus_metrics.append(bytes_per_sector_focus_metric)
    if l1_hit_rate_name:
        l1_hit_rate_focus_metric = FocusMetric(
            l1_hit_rate_name,
            metrics[l1_hit_rate_name].value(),
            NvRules.IFrontend.Severity_SEVERITY_DEFAULT,
            "Try to increase the hit rate in L1TEX to benefit from its higher bandwidth"
        )
        focus_metrics.append(l1_hit_rate_focus_metric)
    if l2_hit_rate_name:
        l2_hit_rate_focus_metric = FocusMetric(
            l2_hit_rate_name,
            metrics[l2_hit_rate_name].value(),
            NvRules.IFrontend.Severity_SEVERITY_DEFAULT,
            "Try to increase the hit rate in L2 to benefit from its higher bandwidth"
        )
        focus_metrics.append(l2_hit_rate_focus_metric)
    if throughput_name:
        throughput_focus_metric = FocusMetric(
            throughput_name,
            metrics[throughput_name].value(),
            NvRules.IFrontend.Severity_SEVERITY_LOW,
            f"The higher the {cache.upper()} throughput the more severe the issue "
            f"becomes"
        )
        focus_metrics.append(throughput_focus_metric)

    return speedup_type, improvement_percent, focus_metrics


def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()
    metrics = RequestedMetricsParser(handle, action).parse(requested_metrics)

    cache_levels = [
        "l1tex",
        "l2",
        "dram",
    ]
    memory_spaces = [
        "global",
        "local",
    ]
    operations = [
        "load",
        "store",
    ]
    threshold_speedup_percent = 0

    # For each memory space/operation combination, store the rule result
    # for the cache level with the highest speedup
    rule_results = {
        "global": {
            "load": RuleResult(),
            "store": RuleResult(),
        },
        "local": {
            "load": RuleResult(),
            "store": RuleResult(),
        },
    }

    # Generate rule messages, speedup estimates and focus metrics
    for cache, space, operation in product(cache_levels, memory_spaces, operations):
        bytes_per_sector, max_bytes_per_sector = \
            get_bytes_per_sector_metrics(metrics, space, operation)

        # Only consider non-trivial loads/stores with less than perfect efficiency
        if 0 < bytes_per_sector < max_bytes_per_sector:
            speedup_type, speedup_value, focus_metrics = \
                get_speedup_and_focus_metrics(cache, space, operation, metrics)

            if speedup_value <= rule_results[space][operation].speedup_value:
                continue

            rule_title = f"{cache.upper()} {space.title()} {operation.title()} Access Pattern"
            cache_level_message = ""
            if cache == "l2":
                if space == "global" and operation == "load":
                    l1_miss_rate = 100 - metrics["l1tex_global_load_hit_rate_percent"].value()
                elif space == "global" and operation == "store":
                    l1_miss_rate = 100 - metrics["l1tex_global_store_hit_rate_percent"].value()
                elif space == "local" and operation == "load":
                    l1_miss_rate = 100 - metrics["l1tex_local_load_hit_rate_percent"].value()
                elif space == "local" and operation == "store":
                    l1_miss_rate = 100 - metrics["l1tex_local_store_hit_rate_percent"].value()
                cache_level_message = (
                    f"This applies to the {l1_miss_rate:.1f}% of sectors missed in L1TEX. "
                )
            elif cache == "dram":
                if operation == "load":
                    l2_miss_rate = 100 - metrics["l2_load_hit_rate_percent"].value()
                elif operation == "store":
                    l2_miss_rate = 100 - metrics["l2_store_hit_rate_percent"].value()
                cache_level_message = (
                    f"This applies to the {l2_miss_rate:.1f}% of sectors missed in L2. "
                )
            rule_message = (
                f"The memory access pattern for {space} {operation}s "
                f"{'from' if operation == 'load' else 'to'} {cache.upper()} "
                f"might not be optimal. "
                f"On average, only {bytes_per_sector:.1f} of the "
                f"{max_bytes_per_sector:.0f} bytes transmitted per sector are utilized "
                f"by each thread. "
                + cache_level_message
                + f"This could possibly be caused by a stride between threads. "
                f"Check the @section:SourceCounters:Source Counters@ section for "
                f"uncoalesced {space} {operation}s."
            )

            rule_results[space][operation] = RuleResult(
                rule_message,
                rule_title,
                speedup_type,
                speedup_value,
                focus_metrics,
            )

    # Send the most impactful rule results to the frontend
    for space, operation in product(memory_spaces, operations):
        if (
            rule_results[space][operation].message
            and rule_results[space][operation].speedup_value >= threshold_speedup_percent
        ):
            message_id = fe.message(
                NvRules.IFrontend.MsgType_MSG_OPTIMIZATION,
                rule_results[space][operation].message,
                rule_results[space][operation].title,
            )
            fe.speedup(
                message_id,
                rule_results[space][operation].speedup_type,
                rule_results[space][operation].speedup_value,
            )
            for metric in rule_results[space][operation].focus_metrics:
                fe.focus_metric(
                    message_id,
                    metric.name,
                    metric.value,
                    metric.severity,
                    metric.advice,
                )
