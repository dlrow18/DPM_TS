from dataclasses import dataclass
from classes.Event import Event

@dataclass
class Constraint:
  predecessor: Event
  successor: Event
  relation: str

  # support: float
  # confidence: float

  def get_predecessor(self):
    return self.predecessor

  def get_successor(self):
    return self.successor

  def get_key(self):
    return (
      str(self.get_predecessor()),
      str(self.get_successor()),
      str(self.relation).strip()
    )

  def __str__(self):
    return str(self.predecessor) + " " + self.relation + " " + str(self.successor)
    # return (f"{str(self.predecessor)} {self.relation} {str(self.successor)} "
    #        f"(sup={self.support}, conf={self.confidence})")