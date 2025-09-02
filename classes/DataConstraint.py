from dataclasses import dataclass
from classes.Event import Event
from typing import List

@dataclass(frozen=True, eq=True)
class DataConstraint:
    predecessor: Event
    successor:   Event
    relation: str
    attribute_name: str
    attribute_values: List[str]

    def matches(self, event: Event) -> bool:
        return str(event.get_attr(self.attribute_name)) == self.attribute_values

    def __str__(self):
        return (f"{str(self.predecessor)} {self.relation} {str(self.successor)} "
                f"{str(self.attribute_name)}")