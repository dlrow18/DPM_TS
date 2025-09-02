from dataclasses import dataclass, field
from classes.State import State
from classes.Event import Event
from classes.Attribute import Attribute
from classes.Transition import Transition
from more_itertools import pairwise

@dataclass
class Trace:
  name: str #can only be used as identifier because we are working with CPEE logs and we therefore have a unique case:concept:name
  events: list = field(default_factory=list)
  prefixes: list = field(default_factory=list)
  
  def __repr__(self):
    return "\n".join((str(ev) for ev in self.events))

  #here trace equality means actually all events in trace are the same and in the same order
  def __eq__(self, other):
    if(len(self.events) != len(other.events)):
      return False
    return all((me == it for me, it in zip(self.events, other.events)))
    
  def get_conceptnames(self):
    res = []
    for ev in self.events:
      if(ev.get_conceptname() is not None):
        res.append(ev.get_conceptname())
    return res
    
  def calculate_prefixes(self, state_abstraction):
    for x in range(1,len(self.events)+1):
      self.prefixes.append(Prefix(self.name, self.events[:x], state_abstraction = state_abstraction))
  
  def calculate_transitions(self, state_abstraction):
    # check if calculate_prefixes has been invoked before to avoid repeating prefixes! only for ats_prediction!
    if self.prefixes == []:
      self.calculate_prefixes(state_abstraction)
    
    abstracted_events = set()
    initial_abstracted_event = self.events[0].apply_event_abstraction()
    abstracted_events.add(initial_abstracted_event)
    
    states = set()
    initial_state = Prefix(self.name, list(), state_abstraction = state_abstraction).state
    states.add(initial_state)
    
    transitions = list()
    transitions.append(Transition(initial_state, initial_abstracted_event, self.prefixes[0].state, True, None))
    
    for prefix_1, prefix_2 in pairwise(self.prefixes):
      state_1 = prefix_1.state
      state_2 = prefix_2.state
      states.add(state_1)
      states.add(state_2)
      abstracted_event = prefix_2.state.abstracted_events[-1]
      abstracted_events.add(abstracted_event)
      transition = Transition(state_1, abstracted_event, state_2, True, None)
      if(transition not in transitions):
        transitions.append(transition)
    return abstracted_events, states, transitions

  
@dataclass
class Prefix(Trace):
  state_abstraction: str = ""
  state: State = None
  
  def __post_init__(self):
    self.state = State(self.state_abstraction, True, None, self.events)
