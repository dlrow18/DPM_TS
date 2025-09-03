import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from collections import defaultdict
import numpy as np
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.algo.filtering.log.variants import variants_filter
import time

from datetime import datetime, timedelta
from pm4py.objects.log.obj import EventLog, Trace as PMTrace, Event as PMEvent
from generate_ts import generate_constraint


# transfer traces in unknwon traces into pm4py event log
def traces_to_eventlog(unknown_traces):
    log = EventLog()
    for tr in unknown_traces:
        pm_trace = PMTrace()
        base_time = datetime.utcnow()
        for idx, ev in enumerate(tr.events):
            event_dict = {attr.key: attr.value for attr in ev.attributes}
            event_dict.setdefault("concept:name", ev.get_conceptname())
            event_dict.setdefault("case:concept:name", tr.name or str(id(tr)))
            event_dict["time:timestamp"] = base_time + timedelta(seconds=idx)
            # if lifecycle exists, add it
            lc = getattr(ev, "get_lifecycle", None)
            if callable(lc):
                lc_val = lc()
                if lc_val:
                    event_dict.setdefault("lifecycle:transition", lc_val)
            pm_trace.append(PMEvent(event_dict))
        log.append(pm_trace)
    return log


# load event log
def load_event_log(file_path):
    df = pd.read_csv(file_path)
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    eventlog = log_converter.apply(df)
    return eventlog


class DeclarativeMiner:
    def __init__(self, min_support=0.01, min_confidence=0.01):
        self.min_support = min_support
        self.min_confidence = min_confidence

        # structures of the knowledge base
        self.activity_freq = defaultdict(int)  # frequency of each activity
        self.trace_presence = defaultdict(int)  # times each activity appears in any trace
        self.directly_follows = defaultdict(lambda: defaultdict(int))  # count of directly follows relationships
        self.eventually_follows = defaultdict(lambda: defaultdict(int))  # count of eventually follows relationships
        self.activity_pairs = set()  # activity pairs that have relationships

        # PM4PY
        self.dfg = None
        self.variants = None
        self.activities = []

    # build knowledge base from event log
    def build_knowledge_base(self, eventlog):
        start_time = time.time()

        # calculate Directly Follows Graph (DFG)
        self.dfg = dfg_discovery.apply(eventlog)

        # retrieve variants from the event log
        variants = case_statistics.get_variant_statistics(eventlog)
        self.variants = variants
        # print(f"Found {len(variants)} Variants")

        # extract all unique activities
        activities = set(act for trace in eventlog for event in trace for act in [event["concept:name"]])
        self.activities = sorted(activities)
        # print(f"Found {len(self.activities)} activities")

        # build knowledge base
        total_traces = 0
        variant_start = time.time()

        for variant in variants:
            variant_count = variant["count"]
            # trace = variant["variant"].split(",")
            trace = list(variant["variant"])
            total_traces += variant_count

            # update activity frequency and trace presence
            unique_acts = set(trace)
            for act in unique_acts:
                self.trace_presence[act] += variant_count
                self.activity_freq[act] += trace.count(act) * variant_count

            # update directly follows relationships
            for i in range(len(trace) - 1):
                a, b = trace[i], trace[i + 1]
                self.directly_follows[a][b] += variant_count
                self.activity_pairs.add((a, b))

            # update eventually follows relationships
            for i, a in enumerate(trace):
                # for each activity, check all subsequent activities
                for j in range(i + 1, len(trace)):
                    b = trace[j]
                    self.eventually_follows[a][b] += variant_count
                    self.activity_pairs.add((a, b))

        self.total_traces = total_traces
        # print(f"knowledge base built | #trace: {total_traces} | #activitiy pairs: {len(self.activity_pairs)} | "
        #      f"time: {time.time() - start_time:.2f}s")

    def calculate_ef_metrics(self, a, b):
        total_traces_with_a = self.trace_presence[a]
        if total_traces_with_a == 0:
            return 0.0, 0.0

        success_count = self.eventually_follows[a].get(b, 0)
        support = success_count / self.total_traces  # in the entire log, how many traces have a followed by b
        confidence = success_count / total_traces_with_a  # in the traces with a, how many have b after a
        return round(support, 4), round(confidence, 4)

    def calculate_df_metrics(self, a, b):
        total_traces_with_a = self.trace_presence[a]
        if total_traces_with_a == 0:
            return 0.0, 0.0

        df_count = self.directly_follows[a].get(b, 0)
        support = df_count / self.total_traces
        confidence = df_count / total_traces_with_a
        return round(support, 4), round(confidence, 4)

    def mine_constraints(self):

        import time
        start_time = time.time()
        constraints = []

        print(f"Start mining new constraints")

        for (a, b) in self.activity_pairs:
            # Eventually Follows
            '''
            resp_sup, resp_conf = self.calculate_ef_metrics(a, b)
            if resp_sup >= self.min_support and resp_conf >= self.min_confidence:
                constraints.append(generate_constraint(a, b, "ev"))
            '''

            # Directly Follows
            succ_sup, succ_conf = self.calculate_df_metrics(a, b)
            if succ_sup >= self.min_support and succ_conf >= self.min_confidence:
                constraints.append(generate_constraint(a, b, "df"))

        print(f"Found {len(constraints)} Constraints | "
              f"Time: {time.time() - start_time:.2f}s"
              f" Mined Constraints: {constraints}")
        return constraints

    def build_from_traces(self, traces):

        eventlog = traces_to_eventlog(traces)
        self.build_knowledge_base(eventlog)

