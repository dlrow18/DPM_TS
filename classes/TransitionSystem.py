from dataclasses import dataclass, field
from graphviz import Digraph
from collections import defaultdict
from classes.State import State
import random

@dataclass
class TransitionSystem:
  states: set = field(default_factory=set)
  events: set = field(default_factory=set)    # list of abstracted events
  transitions: dict = field(default_factory=dict)
  transitions_probabilities: dict = field(default_factory=dict)
  states_annotations: dict = field(default_factory=dict)

  def __repr__(self):
    return "states: " + str(self.states) + ", events: " + str(self.events) + ", transitions: " + str(self.transitions)

  def draw(self, outputpath, filename):
    d_graph = Digraph(name=str(filename), graph_attr={'rankdir':'LR', 'center':'true', 'margin':'1'})
    st = Digraph(name='states', node_attr={'shape':'circle', 'fixedsize':'true', 'height':'.3','width':'.3'})
    for tr,val in self.transitions.items():
      st.node(str(tr.state_1))
      st.node(str(tr.state_2))
      d_graph.edge(str(tr.state_1), str(tr.state_2), label=str(tr.event)+"count"+str(val)+"probab"+str(self.transitions_probabilities[tr]))
    d_graph.subgraph(st)
    d_graph.render(outputpath+"/"+filename, view=False, cleanup=True)
    return outputpath+"/"+filename+".pdf"

  def calculate_probabilities_relative(self):
    counter_per_pre_state = defaultdict(int)
    for transition, count in self.transitions.items():
      counter_per_pre_state[transition.state_1] = counter_per_pre_state[transition.state_1] + count
    transitions_relative_occurence = defaultdict(float)
    for transition,val in self.transitions.items():
      counter = counter_per_pre_state[transition.state_1]
      transitions_relative_occurence[transition] = round(val/counter,2)
    self.transitions_probabilities = transitions_relative_occurence
    # calculate annotations {outgoing_state1: probability, outgoing_state2: probability} for each state
    for state in self.states:
      outgoing_states = {}
      for transition, count in self.transitions.items():
        if transition.state_1 == state:
          outgoing_states.update({transition.state_2: self.transitions_probabilities[transition]})
      self.states_annotations[state] = outgoing_states
  
  # predict for all possible prefixes within a trace
  def predict(self,trace):
    prefixes_prediction = []
    prefixes_true = []
    trace.calculate_prefixes((next(iter(self.states))).state_abstraction)
    for prefix in trace.prefixes:
      current_state = prefix.state
      prefixes_true.append(current_state.abstracted_events[-1].get_conceptname())
      # case 1: unseen state
      if(current_state not in self.states):
        prefixes_prediction.append("unknown")
        continue
      # case 2: no prediction as this is the end state of the transition system
      if self.states_annotations[current_state] == {}:
        prefixes_prediction.append("eoc")
        continue
      # case 3: observed state
      outgoing_states = self.states_annotations[current_state]
      max_prob = max(outgoing_states.values())
      possible_states = list()
      for possible_state, probability in outgoing_states.items():
        if probability == max_prob:
          possible_states.append(possible_state)
        # only one state possible
      if len(possible_states) == 1:
        prefixes_prediction.append(possible_states[0].abstracted_events[-1].get_conceptname())
      else:
        # multiple states possible
        prefixes_prediction.append(random.choice(possible_states).abstracted_events[-1].get_conceptname())
    prefixes_true.append("eoc")
    del prefixes_true[0]
    return prefixes_true, prefixes_prediction

  def online_predict(self,trace):
    trace.calculate_prefixes((next(iter(self.states))).state_abstraction)
    current_state = trace.prefixes[-1].state
    # case 1: unseen state
    if(current_state not in self.states):
      return "Unable to predict as I have never seen this state before."
    # case 2: no prediction as this is the end state of the transition system
    if self.states_annotations[current_state] == {}:
      return "No prediction as this is the end state"
    # case 3: observed state
    outgoing_states = self.states_annotations[current_state]
    max_prob = max(outgoing_states.values())
    possible_states = list()
    for possible_state, probability in outgoing_states.items():
      if probability == max_prob:
        possible_states.append(possible_state)
      # only one state possible
    if len(possible_states) == 1:
      return "Next event lable for "+str(current_state)+" is"+str(possible_states[0].abstracted_events[-1]+" with probability "+str(max_prob))
      # multiple states possible
    res = "MULTIPLE STATES POSSIBLE:"
    for state in possible_states:
      res = res + "\nNext event lable for "+str(current_state)+" is"+str(state.abstracted_events[-1])+" with probability "+str(max_prob)
    return res
      