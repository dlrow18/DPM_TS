from dataclasses import dataclass

@dataclass
class Attribute:
  key: str
  value: str
  
  def __repr__(self):
    return self.key + " " + self.value
    
  def __str__(self):
    return self.value
