from dataclasses import dataclass, field
from classes.Constraint import Constraint

@dataclass
class State:
  state_abstraction: str
  origin_log: bool
  constraint: Constraint
  original_events: list = field(default_factory=list)
  abstracted_events: list = field(default_factory=list)
  
  def __post_init__(self):
    if(self.state_abstraction == "sequence"):
      self.sequence_abstraction()
    elif(self.state_abstraction == "lastevent"):
      self.last_event_abstraction()
    elif(self.state_abstraction == "set"):
      self.set_abstraction()
    else:
      print("no such abstraction found, will use default sequence abstraction")
      self.sequence_abstraction()

  def __repr__(self):
    if(not self.abstracted_events):
      return "{}"
    res = ""
    for event in self.abstracted_events:
      res = res + str(event)
    return res

  def __str__(self):
    if(not self.abstracted_events):
      return "{}"
    res = ""
    for event in self.abstracted_events:
      res = res + str(event)
    return res

  def __eq__(self, other):
    if isinstance(other, State):
      return self.abstracted_events == other.abstracted_events

  def __hash__(self):
    res = ""
    for event in self.abstracted_events:
      res = res + str(event)
    return hash(res)     

  # default implementation for an abstraction function that uses a list of concept names of events
  def sequence_abstraction(self):
    for event in self.original_events:
      abstracted_event = event.apply_event_abstraction()
      self.abstracted_events.append(abstracted_event)
      
  def last_event_abstraction(self):
    if(len(self.original_events) == 0):
      return
    self.abstracted_events.append(self.original_events[-1].apply_event_abstraction())
    
  def set_abstraction(self):
    tmp = set()
    for event in self.original_events:
      abstracted_event = event.apply_event_abstraction()
      tmp.add(abstracted_event)
    self.abstracted_events = list(tmp)
 
  def extend_(self, orig_event, origin_log, constraint):
    original_events = self.original_events.copy()
    original_events.append(orig_event)
    return State(self.state_abstraction, origin_log, constraint, original_events)
