from dataclasses import dataclass
from classes.State import *
from classes.Event import Event

@dataclass
class Transition:
  state_1: State
  event: Event
  state_2: State
  origin_log: bool
  constraint: Constraint = None

  def __repr__(self):
    return "(FROM " + str(self.state_1) + " VIA " + str(self.event) + " TO " + str(self.state_2) + ")"
    
  def __hash__(self):
    return hash(str(self.state_1) + str(self.event) + str(self.state_2))
    
  def return_next_state(self, state):
    if(state == self.state_1):
      return self.state_2
    return None
