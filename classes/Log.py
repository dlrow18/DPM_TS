from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
import csv
import os
from collections import defaultdict
from classes.Attribute import Attribute
from classes.Event import Event
from classes.Trace import Trace
from classes.TransitionSystem import TransitionSystem

@dataclass
class Log:
  input_path: str
  filename: str
  traces: list = field(default_factory=list)
  
  def parse_file_xes(self,lc_filter):
    ns = {'xes': 'http://www.xes-standard.org/'}
    tree = ET.parse(os.path.join(self.input_path, self.filename))
    root = tree.getroot()
    for trace in root.findall('xes:trace', ns):
      events = []
      for event in trace.findall('xes:event', ns):
        attributes = []
        for attribute in event:
          attributes.append(Attribute(attribute.attrib['key'], attribute.attrib['value']))
        events.append(Event(attributes))
      self.traces.append(Trace("todo",events))  
  
  # parsing for CPEE log files; including an option for filtering events based on lifecycle transitions
  def parse_file_csv(self,lc_filter):
    input_file = csv.DictReader(open(os.path.join(self.input_path, self.filename)))
    events = defaultdict(list)
    for event in input_file:
      attributes = list()
      for k,v in event.items():
        attributes.append(Attribute(k,v))
        if(k == "case:concept:name"):
          current_trace = v
      ev = Event(attributes) 
      # filter out calling events !!!
      if(lc_filter[0] == True and ev.get_lifecycle() != lc_filter[1]):
        continue
      events[current_trace].append(ev)
    for k,v in events.items():
      self.traces.append(Trace(k,v))
  
  def create_transition_system(self, state_abstraction):
    label_space = set()
    state_space = set()
    transitions = defaultdict(int)
    for trace in self.traces:
      tr_label_space, tr_state_space, tr_transitions = trace.calculate_transitions(state_abstraction)
      label_space.update(tr_label_space)
      state_space.update(tr_state_space)
      for transition in tr_transitions:
        transitions[transition] = transitions[transition]+1
    return TransitionSystem(state_space, label_space, transitions)
