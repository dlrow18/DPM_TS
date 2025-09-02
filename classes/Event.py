from dataclasses import dataclass, field
from classes.Attribute import *

@dataclass
class Event:
  attributes: list = field(default_factory=list)

  def __repr__(self):
    res = ""
    for attr in self.attributes:
      res = res + str(attr) + " "
    return res
    
  def __str__(self):
    res = ""
    for attr in self.attributes:
      res = res + str(attr) + " "
    return res
   
  def __eq__(self, other):
    if(len(self.attributes) != len(other.attributes)):
      return False
    self_key_set = set()
    other_key_set = set()
    for attr in self.attributes:
     for attr_other in other.attributes:
       if(attr.key == attr_other.key and attr.value != attr_other.value):
         return False
       self_key_set.add(attr.key)
       other_key_set.add(attr_other.key)
    return True

  def __hash__(self):
    res = ""
    for attribute in self.attributes:
      res = res + str(attribute)
    return hash(res)
  
  def get_conceptname(self):
    for attr in self.attributes:
      if(attr.key == "concept:name"):
        return attr.value
    return None

  def get_lifecycle(self):
    for attr in self.attributes:
      if(attr.key == "lifecycle:transition"):
        return attr.value
    return None
    
  # default event abstraction function using just event labels
  def apply_event_abstraction(self):
    return Event([Attribute("concept:name",self.get_conceptname())])
