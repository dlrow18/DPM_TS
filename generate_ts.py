# config_files:
  # synthetic.config: training set (200 cases); test set (200 cases)
  # helpdesk.config: training set (all cases w/o Closed); test set (all cases with Closed)
  # synthetic_dl.config: training set (100 cases); test set (900 cases)
  # helpdesk_dl.config: training set (first 10% cases w/o Closed); test set (rest 90% cases with Closed)
import argparse
import yaml
import os
import csv
from classes.Log import Log
from classes.Constraint import Constraint
from classes.TransitionSystem import TransitionSystem
from classes.Event import Event
from classes.Attribute import Attribute
from classes.Transition import Transition
from classes.AugmentedTransitionSystem import AugmentedTransitionSystem
from classes.DataConstraint import DataConstraint
import pickle
import time
from collections import defaultdict

def generate_data_constraint(pre, suc, rel, attr_key, attr_value):

  pre_event = Event([Attribute("concept:name", pre)])
  suc_event = Event([Attribute("concept:name", suc)])

  attr_value = attr_value.strip() # remove leading and trailing spaces
  if attr_value.startswith('[') and attr_value.endswith(']'):
    attr_value = attr_value[1:-1]
  values = [v.strip() for v in attr_value.split(',') if v]

  return DataConstraint(pre_event, suc_event, rel, attr_key, values)

def read_data_constraints(data_constraint_path, data_constraint_name):
  input_file = csv.DictReader(open(os.path.join(data_constraint_path, data_constraint_name)))
  data_constraints = list()
  for row in input_file:
    data_constraints.append(generate_data_constraint(row['predecessor'],
                                     row['successor'],
                                     row['relation'],
                                     row['attribute'],
                                     row['value']))
  return data_constraints

def generate_constraint(pre, suc, rel):
  pre_attr = [Attribute("concept:name", pre)]
  pre_event = Event(pre_attr)
  suc_attr = [Attribute("concept:name", suc)]
  suc_event = Event(suc_attr)
  return Constraint(pre_event, suc_event, rel)

def read_data(log_path, log_name, constraint_path, constraint_name, fformat, lc_filter):
  log = Log(log_path, log_name)
  if(fformat == "csv"):
    log.parse_file_csv(lc_filter)
  else:
    log.parse_file_xes(lc_filter)
  input_file = csv.DictReader(open(os.path.join(constraint_path, constraint_name)))
  constraints = list()
  for row in input_file:
    constraints.append(generate_constraint(row['predecessor'], row['successor'], row['relation']))
  return log, constraints

def generate_config_dict_from_config_file():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c','--config',help="Path to Config File.")
  args = parser.parse_args()
  with open(args.config, 'r') as stream:
    try:
      conf = yaml.full_load(stream)
    except yaml.YAMLError as exc:
      print(exc)
  config_dict = dict()
  for k,v in conf.items():
    config_dict[k] = v
  return config_dict

def pickle_result(ts, pickle_path):
  picklefile = open(pickle_path, 'wb')
  pickle.dump(ts, picklefile)
  picklefile.close()

def group_constraints_by_relation(constraints):
  result = defaultdict(list)
  for constraint in constraints:
    result[constraint.relation].append(constraint)
  return result
  
def get_next_state(state, transitions):
  res = []
  for transition in transitions.keys():
    next_state = transition.return_next_state(state)
    if(next_state is not None):
      res.append(next_state)
  if len(res) == 0:
    return None
  return res

def extend_beyond_constraint(start, constraint_state, transitions, constraint, extended_states=None, extended_transitions=None):
  if extended_states is None:
    extended_states = set()
    extended_transitions = dict()
  next_event = start.abstracted_events[-1]  
  additional_state = constraint_state.extend_(next_event, False, constraint)
  extended_states.add(additional_state)
  extended_transitions[Transition(constraint_state, next_event, additional_state, False, constraint)] = 1

  next_states = get_next_state(start, transitions)
  if next_states is not None:
    for next in next_states:
      constraint_state = additional_state
      extended_states, extended_transitions = extend_beyond_constraint(next, constraint_state, transitions, constraint, extended_states, extended_transitions)
  return extended_states, extended_transitions


##########################################  
# handle directly follows constraints  
##########################################
def add_df_constraints(constraints, events, states, transitions):
  next_events = set()
  additional_states = set()
  additional_transitions = dict()
  for constraint in constraints:
    pre_abstracted = constraint.predecessor.apply_event_abstraction()
    suc_abstracted = constraint.successor.apply_event_abstraction()
    states = states | additional_states
    transitions = transitions | additional_transitions
    #CASE 1: constraint already in basic TS, do nothing
    if(pre_abstracted in events and suc_abstracted in events):
      print(f"constraint was already observed, do no augmentation for {constraint}")
    #CASE 2: partially seen
    if(pre_abstracted in events and suc_abstracted not in events):
      print(f"partially seen augmentation for {constraint}")
      next_events.add(suc_abstracted)  
      additional_states_c = set()
      additional_transitions_c = dict()
      for state in states:
        if(len(state.abstracted_events) != 0 and pre_abstracted == state.abstracted_events[-1] and state.abstracted_events.count(pre_abstracted) - state.abstracted_events.count(suc_abstracted) == 1):
          next_states = get_next_state(state, transitions)
          constraint_state = state.extend_(constraint.successor, False, constraint)
          additional_states_c.add(constraint_state)
          additional_transitions_c[Transition(state,suc_abstracted,constraint_state, False, constraint)] = 1
          additional_states.add(constraint_state)                                                       
          additional_transitions[Transition(state,suc_abstracted,constraint_state, False, constraint)] = 1
          if next_states is not None:
            for next_state in next_states:  
              additional_states_1, additional_transitions_1 = extend_beyond_constraint(next_state, constraint_state, transitions, constraint)
              additional_states_c.update(additional_states_1)
              additional_transitions_c.update(additional_transitions_1)
              additional_states.update(additional_states_1)
              additional_transitions.update(additional_transitions_1)
      for state in additional_states_c.copy():
        if(len(state.abstracted_events) != 0 and pre_abstracted == state.abstracted_events[-1]): 
          next_states = get_next_state(state, additional_transitions_c)
          constraint_state = state.extend_(constraint.successor, False, constraint)
          additional_states.add(constraint_state)                                                       
          additional_transitions[Transition(state,suc_abstracted,constraint_state, False, constraint)] = 1  
          # extend beyond constraints if further events exist
          if next_states is not None:  
            for next_state in next_states:  
              additional_states_1, additional_transitions_1 = extend_beyond_constraint(next_state, constraint_state, additional_transitions_c, constraint)
              additional_states.update(additional_states_1)
              additional_transitions.update(additional_transitions_1)
    #CASE 3: constraint violation, cannot be handeled as we assume log to be violation free
    if(pre_abstracted not in events and suc_abstracted in events):
      print(f"there must be an error or violation in the log for {constraint}")
  return next_events, additional_states, additional_transitions
##########################################


##########################################  
# handle eventually follows constraints  
##########################################  
def add_ev_constraints(constraints, events, states, transitions):
  next_events = set()
  additional_states = set()
  additional_transitions = dict()
  for constraint in constraints:
    pre_abstracted = constraint.predecessor.apply_event_abstraction()
    suc_abstracted = constraint.successor.apply_event_abstraction()
    states = states | additional_states
    transitions = transitions | additional_transitions
    #CASE 1: constraint already in basic TS, do nothing
    if(pre_abstracted in events and suc_abstracted in events):
      print(f"constraint was already observed, do no augmentation for {constraint}")
    #CASE 2: partially seen
    if(pre_abstracted in events and suc_abstracted not in events):
      print(f"partially seen augmentation for {constraint}")
      next_events.add(suc_abstracted)
      additional_states_c = set()
      additional_transitions_c = dict()
      for state in states:
        if(state.abstracted_events.count(pre_abstracted) > state.abstracted_events.count(suc_abstracted)):  
          next_states = get_next_state(state, transitions)       
          constraint_state = state.extend_(constraint.successor, False, constraint)
          additional_states_c.add(constraint_state)
          additional_transitions_c[Transition(state,suc_abstracted,constraint_state, False, constraint)] = 1
          additional_states.add(constraint_state)
          additional_transitions[Transition(state,suc_abstracted,constraint_state, False, constraint)] = 1    
          if next_states is not None:
            for next_state in next_states:
              # additionally add constraints
              additional_states_1, additional_transitions_1 = extend_beyond_constraint(next_state, constraint_state, transitions, constraint)
              additional_states_c.update(additional_states_1)
              additional_transitions_c.update(additional_transitions_1)           
              additional_states.update(additional_states_1)
              additional_transitions.update(additional_transitions_1)
              
      for state in additional_states_c.copy():
        if(state.abstracted_events.count(pre_abstracted) > state.abstracted_events.count(suc_abstracted)):
          next_states = get_next_state(state, additional_transitions_c)               
          constraint_state = state.extend_(constraint.successor, False, constraint)
          additional_states.add(constraint_state)
          additional_transitions[Transition(state,suc_abstracted,constraint_state, False, constraint)] = 1 
          if next_states is not None:  
            for next_state in next_states:  
              # additionally add constraints
              additional_states_1, additional_transitions_1 = extend_beyond_constraint(next_state, constraint_state, additional_transitions_c, constraint)
              additional_states.update(additional_states_1)
              additional_transitions.update(additional_transitions_1)
              
    #CASE 3: constraint violation, cannot be handeled as we assume log to be violation free
    if(pre_abstracted not in events and suc_abstracted in events):
      print(f"there must be an error or violation in the log for {constraint}")
  return next_events, additional_states, additional_transitions
##########################################


def augment_ts(ts, constraints, data_constraints=None):
  next_events = set()    #abstracted events
  additional_states = set()
  additional_transitions = dict()    #dictionary with counter for transitions occurrences
  constraints_grouped_by_relations = group_constraints_by_relation(constraints)

  # add directly follows constraints
  if(constraints_grouped_by_relations["df"] is not None):
    df_events, df_states, df_transitions = add_df_constraints(constraints_grouped_by_relations["df"], ts.events.union(next_events), ts.states.union(additional_states), ts.transitions | additional_transitions)
    next_events.update(df_events)
    additional_states.update(df_states)
    additional_transitions.update(df_transitions)

  # add eventually follows constraints first
  if (constraints_grouped_by_relations["ev"] is not None):
    ev_events, ev_states, ev_transitions = add_ev_constraints(constraints_grouped_by_relations["ev"], ts.events, ts.states, ts.transitions)
    next_events.update(ev_events)
    additional_states.update(ev_states)
    additional_transitions.update(ev_transitions)
    
  augmented_ts = AugmentedTransitionSystem(ts, constraints, additional_states, next_events, additional_transitions, data_constraints=data_constraints)
  return augmented_ts


def generate_ts(config_dict):  
  log, constraints = read_data(config_dict["log_path"], config_dict["log_name"], config_dict["constraint_path"], config_dict["constraint_name"], config_dict["fformat"], config_dict["lc_filter"])
  data_constraints = read_data_constraints(config_dict["data_constraint_path"], config_dict["data_constraint_name"])

  start = time.time()

  ts = log.create_transition_system(config_dict["state_abstraction"])
  # basic probabilities based on fraction between #(observations for one path from state x)/#(observations of all outgoing paths from state x)
  ts.calculate_probabilities_relative()
    
  image_path_ts = ts.draw(config_dict["output_path"], config_dict["output_name"])
  pickle_path_ts = config_dict["output_path"]+'/'+config_dict["output_name"]+'.pickle'
  pickle_result(ts, pickle_path_ts)

  intermediate = time.time()
  print(f"Time spent for ts construction: {intermediate - start}")
  
  #augmented_ts = augment_ts(ts, constraints)
  augmented_ts = augment_ts(ts, constraints, data_constraints)
  augmented_ts.calculate_probabilities_relative()
  image_path_augmented_ts = augmented_ts.draw(config_dict["output_path"], config_dict["output_name"]+"_augmented")
  pickle_path_augmented_ts = config_dict["output_path"]+'/'+config_dict["output_name"]+'_augmented.pickle'
  pickle_result(augmented_ts, pickle_path_augmented_ts)
  
  end = time.time()
  print(f"Time spent for augmented_ts construction: {end - intermediate}")
  return pickle_path_ts, image_path_ts, pickle_path_augmented_ts, image_path_augmented_ts
  

if __name__ == "__main__":

  config_dict = generate_config_dict_from_config_file()
  generate_ts(config_dict)