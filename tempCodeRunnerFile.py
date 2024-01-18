from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

red = Color
print(red)
print(Color.RED)