from classes.Attribute import Attribute
from classes.Event import Event
from classes.Trace import Trace
from classes.Log import Log
from generate_ts import augment_ts, read_data
from collections import defaultdict
from sklearn import metrics 
import pickle
import argparse
import time
from classes.Miner import DeclarativeMiner
import os, csv

# configuration parameters for mining new constraints
BUFFER_SIZE = 400
MIN_SUP     = 0.7
MIN_CONF    = 0.7
mined_constraints = []


parser = argparse.ArgumentParser(
    description="ATS prediction")

parser.add_argument("--stream_path", 
    type=str, 
    default="data/stream/",   
    help="path to dataset")

parser.add_argument("--stream_name", 
    type=str, 
    default="helpdesk.csv",
    help="dataset name")

parser.add_argument("--constraint_path", 
    type=str, 
    default="data/constraint_patterns/",
    help="path to constraints")

parser.add_argument("--constraint_name", 
    type=str, 
    default="helpdesk_no_resolve.csv",
    help="constraint name")

parser.add_argument("--fformat", 
    type=str, 
    default="csv",   
    help="file format for stream dataset")

parser.add_argument("--lc_filter", 
    type=str, 
    default="[False,""]",
    help="lifecycle filter")

parser.add_argument("--state_abstraction", 
    type=str, 
    default="sequence",   
    help="abstraction for state representation")

parser.add_argument("--ts_model", 
    type=str, 
    default="output/helpdesk_no_resolve.pickle",
    help="path to ts model")

parser.add_argument("--ats_model", 
    type=str, 
    default="output/helpdesk_no_resolve_augmented.pickle",
    help="path to augmented_ts model")

parser.add_argument("--update", 
    type=lambda x: (str(x).lower() == 'true'), 
    default=False, 
    help="update the prediction model")

parser.add_argument(
    "--mined_constraints_file",
    type=str,
    default="output/mined_constraints.csv",
    help="path to save mined constraints"
)

args = parser.parse_args()


# ATS: basic transition system without constraints
def ts_prediction(event_stream, model_path, update=False):
    picklefile = open(model_path, 'rb')
    ts = pickle.load(picklefile)
    picklefile.close()
    y_true = []
    y_pred = []
    for trace in event_stream.traces:
        prefixes_true, prefixes_pred = ts.predict(trace)
        y_true += prefixes_true
        y_pred += prefixes_pred 
        # update transitions as event stream evolves
        if update:
            if prefixes_pred[-1] == "unknown":
                ts = update_ts_unknown(ts, trace)
            else:
                ts = update_ts_known(ts, trace)
        # do not update if unknown behavior occur
        if not update:
            if prefixes_pred[-1] == "unknown":
                ts = ts
            else:
                ts = update_ts_known(ts, trace)

    token_true, token_pred = mapper(y_true, y_pred)
    accuracy = metrics.accuracy_score(token_true, token_pred)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(token_true, token_pred, average="weighted")
    print("##########Prediction results for TS##########")
    print('Accuracy across all prefixes:', accuracy)
    print('F-score across all prefixes:', fscore)
    print('Precision across all prefixes:', precision)
    print('Recall across all prefixes:', recall)
    
# DPMTS: transition system including data aware constraints and online updates
def ats_prediction(event_stream, model_path, update=False):

    unknown_event_buffer = []

    picklefile = open(model_path, 'rb')
    augmented_ts = pickle.load(picklefile)
    picklefile.close()
    y_true = []
    y_pred = []
    has_unknown_event = False

    for trace in event_stream.traces:

        print(".", end="", flush=True)

        prefixes_true, prefixes_pred, has_unknown_event = augmented_ts.predict(trace)
        y_true += prefixes_true
        y_pred += prefixes_pred

        # collect unknown event traces for constraint mining
        if has_unknown_event:
            unknown_event_buffer.append(trace)

        # if enough unknown events are collected, mine new constraints
        if len(unknown_event_buffer) >= BUFFER_SIZE:
            '''
            # print some examples of unknown traces in buffer firstly
            print(f"\n[INFO] Buffer reached {len(unknown_event_buffer)} unknown traces. Showing a few examples:\n")

            for i, tr in enumerate(unknown_event_buffer[:3]):
                event_names = [e.get_conceptname() for e in tr.events]
                print(f"  Trace {i + 1} (case={tr.name}): {' → '.join(event_names)}")
            '''
            print(f"\n[INFO] Buffer reached {len(unknown_event_buffer)} unknown traces. Start mining new constraints...")

            # mine new constraints from the unknown traces
            miner = DeclarativeMiner(MIN_SUP, MIN_CONF)
            miner.build_from_traces(unknown_event_buffer)
            all_constraints = miner.mine_constraints()

            existing_keys = {c.get_key() for c in mined_constraints}
            new_constraints = [c for c in all_constraints if c.get_key() not in existing_keys]

            mined_constraints.extend(new_constraints)

            # generate new augmented transition system with all constraints
            augmented_ts = update_constraints(augmented_ts.transition_system, new_constraints)
            augmented_ts.calculate_probabilities_relative()

            unknown_event_buffer.clear()

        # update transitions as event stream evolves
        if update:
            if prefixes_pred[-1] == "unknown":
                augmented_ts = update_ats_unknown(augmented_ts, trace, augmented_ts.constraints)  # original constraints
            else:
                augmented_ts = update_ats_known(augmented_ts, trace)
        # do not update if unknown behavior occur
        if not update:
            #if prefixes_pred[-1] == "unknown":
            augmented_ts = augmented_ts
            #else:
            #    augmented_ts = update_ats_known(augmented_ts, trace)


    token_true, token_pred = mapper(y_true, y_pred)
    accuracy = metrics.accuracy_score(token_true, token_pred)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(token_true, token_pred, average="weighted")
    print("##########Prediction results for DPMTS##########")
    print('Accuracy across all prefixes:', accuracy)
    print('F-score across all prefixes:', fscore)
    print('Precision across all prefixes:', precision)
    print('Recall across all prefixes:', recall)

def mapper(y_true, y_pred):
    token_true = []
    token_pred = []
    keys = set(y_true) | set(y_pred)
    val = range(len(keys))
    labels_dict = dict(zip(keys, val))
    for _y in y_true:
        token_true.append(labels_dict[_y])
    for _y in y_pred:
        token_pred.append(labels_dict[_y])
    return token_true, token_pred

# update transitions for ts if trace has been observed before 
def update_ts_known(ts, trace):
    # update according to involving stream regardless of predicted value is true or false
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    for transition in tr_transitions:
        ts.transitions[transition] += 1
    ts.calculate_probabilities_relative()
    return ts

# update transitions for ts if trace has not been observed before 
def update_ts_unknown(ts, trace):
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    ts.events.update(tr_label_space)
    ts.states.update(tr_state_space)
    for transition in tr_transitions:
        ts.transitions[transition] += 1
    ts.calculate_probabilities_relative()
    return ts

# update transitions for augmented_ts if trace has been observed before 
def update_ats_known(augmented_ts, trace):
    # update according to involving stream regardless of predicted value is true or false
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    for transition in tr_transitions:
        if transition in augmented_ts.transition_system.transitions:
            augmented_ts.transition_system.transitions[transition] += 1
        if transition in augmented_ts.additional_transitions:
            tr_constraint = None
            for constraint in augmented_ts.constraints:
                predecessor = constraint.predecessor.apply_event_abstraction()
                successor = constraint.successor.apply_event_abstraction()
                pre_state = transition.state_1.abstracted_events
                suc_state = transition.state_2.abstracted_events  
                if predecessor in pre_state and successor not in pre_state and successor in suc_state:
                    tr_constraint = constraint
            transition.origin_log = False
            transition.constraint = tr_constraint
            augmented_ts.additional_transitions[transition] += 1
    augmented_ts.calculate_probabilities_relative()
    return augmented_ts

# update transitions for augmented_ts if trace has not been observed before 
def update_ats_unknown(augmented_ts, trace, constraints):
    tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(args.state_abstraction)
    # update the basic transition system
    updated_ts = augmented_ts.transition_system
    updated_ts.events.update(tr_label_space)
    updated_ts.states.update(tr_state_space)
    for transition in tr_transitions:
        updated_ts.transitions[transition] += 1
    # augment the updated transition system with original constraints
    new_ats = update_constraints(updated_ts, constraints)
    new_ats.calculate_probabilities_relative()
    return new_ats

def update_constraints(ts, constraints):
    augmented_ts = augment_ts(ts, constraints)
    return augmented_ts

def event_name(e):
    return e.get_conceptname() if hasattr(e, "get_conceptname") else str(e)

def write_constraints(filepath: str):

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["predecessor", "successor", "relation"])
        w.writeheader()
        for c in mined_constraints:
            w.writerow({
                "predecessor": event_name(c.predecessor),
                "successor":   event_name(c.successor),
                "relation":    str(c.relation).strip(),
            })

if __name__ == "__main__":
    # ATS without constraints: no updates
    stream_ts, constraints = read_data(args.stream_path, args.stream_name, args.constraint_path, 
                                    args.constraint_name, args.fformat, args.lc_filter)  
    ts_prediction(stream_ts, args.ts_model, args.update)

    # AATS including constraints: no updates
    stream_ats, constraints = read_data(args.stream_path, args.stream_name, args.constraint_path, 
                                    args.constraint_name, args.fformat, args.lc_filter)
    start = time.time()
    ats_prediction(stream_ats, args.ats_model, args.update)

    write_constraints(args.mined_constraints_file)
    
    end = time.time()
    print(f"##########Retrain and prediction time for DPMTS: {end-start}##########")