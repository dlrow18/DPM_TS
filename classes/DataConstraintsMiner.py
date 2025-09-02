from __future__ import annotations

import math
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import re

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

# Module-level configuration flags (lowercase: they are not constants in logic)
split_or_constraints = True    # split OR into multiple constraints
simplify_conditions  = True    # compress redundant thresholds
reverse_mapping      = True    # map numeric split points back to categorical labels

# Column name constants (kept uppercase as stable config)
CASE_COL = "case:concept:name"
ACT_COL  = "concept:name"
TIME_COL = "time:timestamp"

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]]]:
    """Label-encode object/category columns; return encoded frame + code→label maps."""
    encoded = df.copy()
    category_maps: Dict[str, Dict[int, str]] = {}
    for col in encoded.select_dtypes(include=["object", "category"]).columns:
        cat = encoded[col].astype("category")
        encoded[col] = cat.cat.codes
        category_maps[col] = dict(enumerate(cat.cat.categories))
    return encoded, category_maps

# Precompile a regex for atomic numeric predicates like: "amount <= 3.5".
# Captures: (attr) (op) (value) — e.g., "age >= 18", "score < 0.7"
atomic_predicate_pattern = re.compile(r"(\w+)\s*([<>]=?)\s*([0-9.]+)")


def simplify_path(path: str) -> str:
    """Collapse redundant <= / > thresholds on the same attribute into tight bounds."""
    by_attribute: Dict[str, Dict[str, float]] = defaultdict(lambda: {"<=": math.inf, ">": -math.inf})
    for cond in path.split(" & "):
        match = atomic_predicate_pattern.match(cond.strip())
        if not match:
            continue
        attr, op, val = match.groups()
        val = float(val)
        if op.startswith("<"):
            by_attribute[attr]["<="] = min(by_attribute[attr]["<="], val)
        else:
            by_attribute[attr][">"] = max(by_attribute[attr][">"], val)
    parts: List[str] = []
    for attr, bounds in by_attribute.items():
        if bounds[">"] > -math.inf:
            parts.append(f"{attr} > {bounds['>']:.4g}")
        if bounds["<="] < math.inf:
            parts.append(f"{attr} <= {bounds['<=']:.4g}")
    return " & ".join(parts)


def reverse_map_condition(condition: str, category_maps: Dict[str, Dict[int, str]]) -> str:
    # Convert numeric thresholds back to "attr in {v1, v2}" for categorical attributes.

    def replace_numeric_threshold(match: re.Match) -> str:
        attr, op, val = match.groups()
        val = float(val)
        if attr not in category_maps:
            # numeric attribute: keep original numeric predicate
            return match.group(0)
        code_to_label = category_maps[attr]
        code_ids = list(code_to_label.keys())
        if op.startswith("<"):
            upper = int(math.floor(val))
            labels = [code_to_label[i] for i in code_ids if i <= upper]
        else:  # >
            lower = int(math.floor(val))
            labels = [code_to_label[i] for i in code_ids if i > lower]
        label_str = ", ".join(map(str, labels)) or "<none>"
        return f"{attr} in {{{label_str}}}"

    return atomic_predicate_pattern.sub(replace_numeric_threshold, condition)


def learn_conditions(features_df: pd.DataFrame, labels: List[int], *, max_depth: int = 3) -> List[str]:
    # Return one condition string per positive leaf (already simplified/mapped).
    encoded_df, category_maps = encode_categorical(features_df)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5, random_state=42)
    clf.fit(encoded_df, labels)

    conditions_out: List[str] = []
    feature_names = encoded_df.columns
    tree_struct = clf.tree_

    def collect_paths_dfs(node: int, conditions: List[str]) -> None:
        if tree_struct.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_struct.feature[node]]
            thr = tree_struct.threshold[node]
            collect_paths_dfs(tree_struct.children_left[node],  conditions + [f"{name} <= {thr:.4g}"])
            collect_paths_dfs(tree_struct.children_right[node], conditions + [f"{name} > {thr:.4g}"])
        else:
            predicted_class = tree_struct.value[node][0].argmax()
            if predicted_class == 1:  # fulfil
                path_text = " & ".join(conditions)
                if simplify_conditions:
                    path_text = simplify_path(path_text)
                if reverse_mapping:
                    path_text = reverse_map_condition(path_text, category_maps)
                conditions_out.append(path_text)

    collect_paths_dfs(0, [])
    return conditions_out

def mine_constraints(df: pd.DataFrame, *, business_cols: List[str], min_sup: float,
                     min_fulfill: float, max_depth: int) -> pd.DataFrame:
    # mine Response[A,B] constraints with data conditions.
    # build traces (case → list of event dicts)
    traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, row in df.iterrows():
        traces[row[CASE_COL]].append(row.to_dict())

    activity_set = set(df[ACT_COL])
    candidate_pairs = [(a, b) for a in activity_set for b in activity_set if a != b]

    results: List[Dict[str, Any]] = []
    total_events = len(df)
    min_activations_required = max(1, int(min_sup * total_events))

    for pre_event, post_event in candidate_pairs:
        activation_snapshots: List[Dict[str, Any]] = []
        activation_labels: List[int] = []
        for events in traces.values():
            activation_indices = [i for i, e in enumerate(events) if e[ACT_COL] == pre_event]
            for idx in activation_indices:
                snapshot = {col: events[idx].get(col) for col in business_cols}
                fulfilled = any(e[ACT_COL] == post_event for e in events[idx + 1:])
                activation_snapshots.append(snapshot)
                activation_labels.append(int(fulfilled))
        if len(activation_snapshots) < min_activations_required:
            continue
        fulfillment_ratio = sum(activation_labels) / len(activation_labels)
        support_ratio = len(activation_snapshots) / total_events
        if fulfillment_ratio < min_fulfill:
            continue
        features_df = pd.DataFrame(activation_snapshots)
        condition_list = learn_conditions(features_df, activation_labels, max_depth=max_depth)
        for condition_text in condition_list:
            results.append({
                "rule": f"Response[{pre_event}, {post_event}]",
                "activation_cond": condition_text,
                "support": support_ratio,
                "fulfil_ratio": fulfillment_ratio,
            })
    return pd.DataFrame(results)


def main(args) -> None:
    df = pd.read_csv(args.csv)
    # clean column names (strip spaces/BOM)
    df.columns = df.columns.str.strip().str.replace(r"^\ufeff", "", regex=True)
    if {CASE_COL, ACT_COL, TIME_COL} - set(df.columns):
        raise ValueError("Basic column missing（case/activity/time）")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values([CASE_COL, TIME_COL])

    business_cols = args.cols
    result = mine_constraints(df,
                              business_cols=business_cols,
                              min_sup=args.min_sup,
                              min_fulfill=args.min_fulfill,
                              max_depth=args.max_depth)
    out_dir = Path("../output"); out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "data_constraints.csv"
    result.to_csv(out_file, index=False)
    print(f"Saved {len(result)} constraints → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=Path(__file__).parents[1] / "data" / "logs" / "helpdesk_no_resolve.csv")
    parser.add_argument("--cols", nargs="+", default=["workgroup"])
    parser.add_argument("--min-sup", type=float, default=0.1)
    parser.add_argument("--min-fulfill", type=float, default=0.9)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()
    main(args)
