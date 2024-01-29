#python by Guidio Van Rossum
#interpreting language

#Values and data types
x = None
y = True and False #Boolean 
z = -1 #integer 
a = .1 #float
b = 2 + 3j #complex
c = 'hey' #string
d = ('hi', True, 1) #tuple
e = ['hi', False, .01] #list
f = set(1, None, 'no') #set
g = {'a':None, True:1} #dictionary
h = b'\n\x14\x1e(2' #bytes
#mutable >> idependent class(index changing occurs) -> data structure using multiple addresses
#immutable >> index replacing is impossible, but function might work, data structure in a single address

#Variables >> numbers letters and underscores 
x, y
##Operators 
#Arithmatic 
1 + 3
2 - 2
3 * 4 
7 / 2 
5 % 2 #divide and returns reminder 
4 ** 3 #exponetial 
-11 // 3 #floor division 
#Relational 
a == b #is
a != b #isnot
>
<
>=
<=
#Assignment
= 
#In place operations
+= #a += b >> a = a + b 
-= 
*=
/=
%=
**=
//=
#Logical 
and 
or
not
#Operator Precedence
** 
~,+,- #complement and unary plus and minus, ~: negation of bit number
*,/,%,//
+, - #operators
a & b #binary multiplication 
a | b #binary summation with carry
a ^ b #binary summation without carry
>>, << #bit shifting  
<=, >=, <, >
==, != 
=, +=, -=, /=, %=, **=, //= 
is, is not #identity
in, not in #membership
not, or, and #logical
#Reserved words -> grammatical words
assert #checks the conditions, continues if TRUE, or does not if FALSE
del #deletes following object
pass #passes the logic

##Python Data Structures
#Iterables
#List >> mutable
list() #only one object available 
[] #or this
#indexing and slicing are similar 
a = list('cat')
#positive numbers are from left to right. negative numbers are from right to left. starting at 0 and -1 excluding.
print(a[0:2], a[2], sep ='\n')
#multi-dimension is also possible 
x = [[1,2],[3,4]]
print(x[0][1])
#updating lists
x[0] = [0,2]
print(x)
#list operations
[] + []
[] * int()
'a' in [] = True or False 
#list functions 
len(x)
del x[slice]
x.append(value)
x.extend(list)
x.pop(index)
x.remove(value)
x.count(value)
x.insert(index, value)
x.index(value) #states the index of 
x.sort()
x.reverse()
x.copy() 
#copying list
#use .copy() or [:] #numpy and c++ case of deep copy
#Stack using list 
list()
#use these operation only
append() #O(1) for python list
pop() #O(1) 

#Tuples -> immutable, addresses in a single address
#it is good that where the addresses are stored does not matter
(,) #indexing and slicing 
tuple()
#packing and unpacking
a = ('a', 'b', 'c')
x,y,z = a
print(x,y,z, sep = '\t') #print(x) will result 'a'
#changing the data in tuples are impossible but mutable data in tuples can change
a = ([1,2],)
a[0][1] = 'p'
#tuple operations
() + ()
() * int()
'a' in () = True or False 
#tuple functions 
len(x)
max(x)
min(x)
x.index(value)
x.count(value)

#Dictionary
#mutable and unordered
{}
dict()
#it can use an iterable of two-item iterables, or an iterable of two character strings as input
dict(a = 4, b = 3)
a = {'a' : 4, 'b' : 3}
#accessing value in dictionary
a = ('12', '24', '56') #keys cannot be repeated. -> Key:Value
d = dict(a)
d['1']
#adding new key and value
d['new key'] = 'value'
#operation and functions 
'2' in d #only key works 
len(d)
del d[key]
d.clear()
d.items() #makes a dictionary into a tuple
d.keys()
d.values()
d.update(dict2) #adds dict2
d.get(key,default) #getting the value of a key, also default value can be set when key is not around
tuple(d.values())
#coping dictionary 
a = d.copy()
del a['1']
a
d

#Set -> unique, unordered elements 
{}
set() #only with lists, strings, tuples and dictionaries (only with keys).
#empty set -> variable = set(), {} does not work 
#unordered -> slicing and indexing is impossible
#operations 
a = set('cat'); b = set([1,2])
'c' in a  
0 not in b
a == b
a != a
a <= a #subset
a >= b #superset
a - b 
a & b #intersection
a | b #union
a ^ b #aUb - a&b
#set functions 
len(x)
x.add(element)
x.update(element)
x.discard(element)
x.remove(element)
x.clear()
#coping set
a 
c = a.copy() #new address assigned 
c.add('p')
c
a
#enum library -> enumeration class
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

red = Color.RED
print(red.value)
print(red.name)

#Built-in functions -> already build in the language, C is used
#see myfunctionbuilding.py 
sum(iterable) #adds inputs 
any(boolean iterable) #returns or logic 
all(boolean iterable) #returns and logic
max(iterable) #biggest 
min(iterable) #smallest
zip(iterable1, iterable2) #simultaneously return multiple iterables, discards when length diffrence occurs  
enumerate(iterable) #counting the order, starting index can be changed
map(function, iterable) #returns the range of the function from iterable domain 
reversed(sequence) #reverses iterable
isinstance(object, type) #check the type of the object
#returns the element in iterable if the result of the function is true when the input was the element
filter(function, iterable)
#print a variable, file -> writing method, flush -> sees whether the outcome is true or false.
type(c) #print the value type of the variable 
#Mathematical functions, built in 
abs()
round(x, n) #type(x) = float, n = nth round up
sorted(iterable, key, reverse) #returns the sorted iterable, key reference, reverse = True or False
#Methods -> methods apply only for a certain class(libraries) -> look up oop, built with Python
#Math module -> built-in standard library
import math
math.ceil(x) #round up
math.factorial(x)
math.floor(x)
math.isfinite(x)
math.isinf(x)
math.isnan(x)
math.trunc(x)
math.exp(x)
math.log(x, 10) #(x,base)
math.log10(x)
math.sqrt(x)
math.cos(x)
math.sin(x)
math.tan(x)
math.degrees(x) #radian to degree
math.pi 
math.e

##Containers -> also called generic data structures
#Strings
str(x) #makes characters in the function a string
int(x) #make the string to an integer
repr(x) #returns the string representation of the object
a = 1 
b = "1"
print("a =" a, "b =" b, sep = '\n')
#this will not distinguish the type of the variable
repr(a) 
repr(b) #this will distinguish the type of the variable\
#Indexing >> gives an individual character an index
#uses Unicode(IT Foundation)
word = 'python'
word[3]
#Slicing >> to obtain substring
word[0:2] #does not include the last index
word[::3] #step(starts from 0)
word[3::-1] #default is [0, the last index + 1, 1]
L = len(word) #shows the number of characters in the string
#Methods
x.find(str) #determines str is in x or not 
x.count(str) #counts how many times does str repeats in x 
x.lower()
x.upper()
x.capitalize() #first letter becomes uppercase
x.title() #make the string a title
x.split(str) #deletes str in x and splits x 
x.replace(old, new) #relaces old characters in the string to new 
x.startswith(str) #determines whether x begins with str
x.endswith(str) #vice a versa 
x.isalpha() #does x contain only alphabet
x.isdigit() #does x contain only digit 
x.isupper() #does x contain only uppercase
x.islower() #does x contain only lowercase
x.isnumeric() #does x contain only numeric value
x.isalnum() #does x contain both numeric and alphabet 
x.strip() #gets rid of spaces on left and right 
x.lstrip() #gets rid of space on the left
x.rstrip() # gets rid of psace on the right
x.count(pattern) #counts how many times the pattern is repeated in x
x.rfind(pattern) #gives the index of the last appearance of the pattern
x.join(iterable) #joins strings in iterable with x appearing each joining 

a = 'a' + 'b' #this is possible. One can add variables between strings. 

#String is immutable
#escape words
'\n' #next line
'\r' #carriage return and inserts 
'\t' #tab
'\b' #back space
'\\' #back slash
'\'' #apostrophex
'\"' #qoute mark
'\e' #esc key
r'string\n' #this will not apply escape words 
'sfsf' in 'sfsf2' #can check
#Regular Expression -> advanced pattern finding
#pattern defining 
'\w' #english letter and digit, also under bar is included
'\W' #not letter or digit, esacpe word etc..
'\d' #digit
'\D' #words other than digit 
'\s' #escape words 
'\S' #other than escape words 
'\ba' #starting or ending with a 
x.find('book\s\d') #book 1, book 2, etc..
#meta character: ^, $, *, +, ?, {}, [], \, |, (), ., :
#add \ in front to use them in expression
#these chracters are used to express regular expression
'/[adsfa]/' #return any word that include more than one letter in []
'/[0-3]/' #- is used to express range 
'/[^abd]/' #return words that does not have these words 
'/.a./' #use dot for any word
'/ab*c/' #returns words that has more than 0 time repetition ex: ac, abc, abbc, ...
'/ab+c/' #more than 1 time  
'/ab?c/' #only once or 0 time ex: abc or ac
'/ab{m,n}c/' #minimum m and maximum n repetition
'/ab{m,}c/' #maxumum m 
'/ab{,n}c/' #minimum n 
'/ab{m}c/' #exactly m ex:\d{4}
'/a|b/' #a pattern or b pattern
'/^the/' #beginning of a line
'/\.$/' #end of the line 
'/(\w{2})(\d{3:2})\1\2/' #prioritizes, captures and saves the parenthesis expression
'/(?:\w{3})' #does not use capturing but only prioritize 
'/...(?=pattern)/' #ends with the pattern but returns in front only
'/...(?<=pattern)/' #starts with the pattern but returns only the rear 

#create pattern with raw string
pattern = '\d{3}-\d{4}-\d{4}'
#Output function
print(1,2,3,4 ,sep='*', end=' hey')
#1*2*3*4 hey
#% formatting
'%d' #works like an integer variable in a string
print('%d hi' %(4))
#naming
print('%(name)s %(age)d' %{'name':'jack','age':24})
'%s' #works like a string variable in a string 
'%f' #works like a float varialbe in a string
'%o' #works like a octaldigit in a string
'%x' #works like a hexadigit in a string
#the letters between % and s or d manipulates the input
#this can be done also with the formatting methods above
#padding
#number between % and letter keeps the word length 
print('%4d + %5d' %(1,2)) #puts the spacing in front 
print('%-4d + %-5d' %(1,2)) #puts the spacing after
print('%04d + %05d' %(1,2)) #puts zero rather than spaces
print('%.2f + %.4f' %(1.3334, 2.334)) #manages the floating digit number
print('%3.2f + %7.4f' %(1.3334, 2.334)) #manages the spacing with floating 
print('%03.2f + %07.4f' %(1.3334, 2.334)) #also changing spaces to zero is possible
#method formatting
x = 3; y = 5 #all the operations must be seperated in line or semicolon
print('I love {} and {}'.format(x,y)) #values assigned come out
print('I love {1} and {0}'.format('bread','butter')) #the numbers in {} assigns output values' order
print('{1:8s}{0:7s}'.format('bread','butter')) #padding can be applied
#naming
print('{cat:8s}{dog:7s}'.format(dog = 'Scooby', cat = 'Tom'))
#f formatting
print(f'I love {x} and {y}') #best way for now 
print(f'I love {x:5d} and {y:6d}') #padding is available
#input function >> adds a step before an assignment 
x = input('how are you?:')

##Control-flows # - comment, \ - line continues, tab - indentation
#Iterator: a code, function that loads multiple data 
#loaded data can be assigned in many addresses or as in generators
#Generator: a code, function that loads multiple data using single address -> loaded data is replaced for each load 
*args #-> creates an interator from function inputs and also from the elements of iterables as an input
def iterate(*args):
    for i in args:
        print(i)
iterate(1,2,3) #pack inputs and create an iterable
list = [1,2,3]
def print_list(args):
    print(*args) #unpacked args returns an iterable
print_list(list) #-> uses the iterable and prints elements only 
print(list) #-> prints the list data structure 
**kwargs #-> packing and unpacking dictionaries
def make_dictionary(**kwargs): 
    print(kwargs)
make_dictionary(x = 1, y = 4) #takes variable assignment and makes a hash relation and dictionary
dict1 = {'x': 1, 'y': 4} 
dict2 = dict(a = 3, y = 'red')
def print_dictionary(kwargs1, kwargs2):
#unpacked kwargs using ** returns hash relation and becomes a dictionary when it is in {}.      
    print({**kwargs1, **kwargs2}) #merges two has relations, overlapping key gets the value of latter variable
print_dictionary(dict1, dict2)
yield # a reserved word that uses single memory address in a generator
return # a reserved word that stores loaded data in different addresses

#if conditionals
if condition:
    statement

exam1 = 90; exam2 = 85

if exam1 >= 100:
    print('Good job! Jake')
elif exam2 >= 80: #the next condition. else if. 
    print('Good job! Joe')
else:
    print('Good!')

#when upper condition satisfied, the statement executes. Then, elif. Then, else. 

#for loop 
for iterating_var in iterable:
    statements 

a = 'key'

for i in 'key':
    print('key\'s characters are: ', i, sep = '\n')

range(a,b,i) #a list of integers that a <= x < b, seperated by i, a == 0 default. i == 1 

for i in range(len(a)):
    print('hihi:', a[i])

count = 0

for i in range(10):
    for j in range(10):
        count += j

x = range(0,10,2)
for i, j in enumerate(x): #enumerate() returns (index, value) of an iteratable
    print('index:', i)
    print('value:', j)

x = [(1,2,3)]
for i, j, k in x: #mutiple iterating_var
    print(k,j,i)

#while loop 
while condition: #must be true and not repeating, otherwise, does not work or infinite loop
    statement 

maxsum = 50 
sum = 0 
i = 0 
while sum <= maxsum:
    i += 1 
    sum += i
print('The sequence is:', list(range(1, i + 1)))

#Loop control
break #a reserved that terminates the loop and transfers to the next loop 
#use / to continue the code to the next line

var = 10 
while var > 0:
    print('Current variable value:', var)
    var -= 1
    if var == 5: 
        break

continue #passes a control flow, a reserved word

for i in 'python':
    if i == 'p':
        continue 
    print('hi')

#list comprehension >> using logical expressions to build lists
#other containers can be built with comprehension 

[(x,y) for x in range(3) for y in range(4) if x + y < 3]

x = list(range(-5,5))
z_set = {x[i]**2 for i in range(len(x))}

z_tuple = tuple(i**3 for i in x)

word = 'letters'
letter_counts = {letter:word.count(letter) for letter in set(word)}

def function(var): #creating one's own function. Variable can follow order or assigned (var = x).
    argument
#variables are in distinguished local memory for functions 
global #use this reserved word to access variables outside the function
nonlocal #one upper level variable access 
var = 0 #global
def main():
    var = 0 #local, the change of this varible does not affect global var 
    global var #now this changes global var 
    def sub():
        var = 0 #this also does not affect global or upper local var 
        nonlocal var #now this changes upper local var in main()
        break
#in making the code clear, using these words are not recommanded
#variable capture -> functions that makes their global variables change the function's result 
def add_one(x):
    global var 
    return x + var 
var = 1
print(add_one(3)) # but this function will change the result when global changes 
var = 2 
print(add_one(3)) #so this is a variable capture case
#closure function -> functions without variable capture, often has no input variable, works on their own
#factory function -> closure function generator, makes a function a variable object 
def print_closure_factory(number): #creating closure factory
    def print_closure(): #code for closure function 
        print(number)
    return print_closure #returning closure function object
print_5 = print_closure_factory(5) #assigning object as a closure function with local variable
print_10 = print_closure_factory(10)
print_5()
print_10() #closure functions
#application of closure objects 
def add(var): #input objects 
    return var + 2
def multiply(var):
    return var * 2
def factory(function, n, var): #input object as functions 
    def closure():
        for i in range(n):
            nonlocal var #brings var from factory variables
            var = function(var)
        return print(var)
    return closure 
add_2_4_times_to_10 = factory(add, 4, 10)
multiply_2_5_times_to_20 = factory(multiply, 5, 20)
add_2_4_times_to_10()
multiply_2_5_times_to_20()
#decorator -> a function using a single input as a function, the decorator will return new function with original input
def print_decorator(function): #function as the input
    def print_closure(var): #has the original input
        print('Input:', var)
        out = function(var)
        print('Output:', out)
    return print_closure
@print_decorator #using decorator to put more features to the original function 
def divide(var): #now divide will also print the input and output
    return var / 2 
divide(2)
#wrap the decorator to add input to it 
def n_times_decorator_factory(times):
    def n_times_decorator(function):    
        def closure(var):
            for i in range(times):
                var = function(var)
            return var
        return closure
    return n_times_decorator
@n_times_decorator_factory(5)
def sub(var):
    return var - 2
print(sub(10))
#since decorating cannot access the original function, use built in library 
from functools import wraps
def print_decorator(function):
    @wraps(function)
    def print_closure(var):
        print('Input:', var)
        out = function(var)
        print('Output:', out)
    return print_closure
@print_decorator
def divide(var):
    return var / 2 
print(divide.__name__) #this will return 'print_closure' if Wraps is not used
#recursive function -> uses itself repeatedly, loops can generate the same code 
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1) #returns the result of itself 
#function parameters 
def function(var1, var2):
    print(var1, var2)
function('a', 2) #positional parameter
function(var2 = 2, var1 = 'a') #naming parameters 
#default parameters can be set, but they must come after free parameters 
def function(var1, var2 = 'hi'): #def function(var2 = 'hi', var1) -> this will make an error 
    print(var1, var2)
#packing inputs in defining a function
def function(*args):
    j = 0
    for i in args:
        j += i
    return j
print(function(1,2,3,4))
#key inputs are possible 
def function(**kwargs):
    print(kwargs)
function(dogs = 4, men = 'dogs')
#parameter type hints, input: types -> output type: 
def function(var1: type) -> type:
    pass
print(function.__annotations__) #can be seen with this method 
#unlike c++, this annotation won't cause an error even if the input is not the type, but many associate programs utilize this
#many code editors also use this to give hints (put dot then ctrl + space for vscode, automatic error check)
#one can add more detailed annotation using typing package
from typing import Annotated

def say_hello(name: Annotated[str, "this is just metadata"]) -> str:
    return f"Hello {name}" #one can access the function's detailed annotation now
#generic annotation 
from typing import List, Dict, Union, Optional
def function(var1: List[str]) -> List:
    for i in var1:
        i.capitalize() #the editor will know i is a string
def function(var1: dict[str, int]) -> List:
    for item_name, item_price in dict.items():
        print(item_name, item_price) #the editor will know item_name is a string and item_price is an integer
#multi type generic
def function(var1: Union[str, int]) -> List:
    pass
#or this, now the editor will know var1 is either string or integer
def function(var1: str | int) -> List:
    pass
#optional type
#using Union but conditional
#string or None case
def function(var1: Optional[str] = None):
    if var1 is not None:
        print(var1.capitalize())
    else:
        print("type is wrong")
#also this is possible 
def function(var1: str | None = None):
    pass
#using pydantic for automation of type annotation
#for a dictionary or json format 
from datetime import datetime
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str = "John Doe"
    signup_ts: datetime | None = None
    friends: List[int] = []

external_data = {
    "id": "123",
    "signup_ts": "2017-06-01 12:22",
    "friends": [1, "2", b"3"], #b"3" will be converted to 3
}
user = User(**external_data) #the package will automatically change the type of the input
#also retracting the data is possible
user_dict = user.dict() #keyword arguments of the model instance

#asyncio for asynchronous programming like in the multiprocessing repository
import asyncio

async def say_after(delay, what): #async function
    await asyncio.sleep(delay)
    print(what)

async def main():
    print(f"started at {time.strftime('%X')}")

    await say_after(1, 'hello') #just like the future like library of c++
    await say_after(2, 'world')

    print(f"finished at {time.strftime('%X')}")

asyncio.run(main()) #or run 

#python hacks 
#comprehension 
result = [i for i in range(n)] #faster than for + append method
result = (i for i in range(n)) #tuples are possible
result = {str(i):i for i in range(n)} #dictionary version 
result = {str(i) for i in range(n)} #set comprehension
#adding conditions 
result = [i for i in range(n) if i % 2 == 0] #if condition
result = [(i,j) for i in range(n) for j in range(i)] #multiple for loops
I = [[int(i == j) for i in range(n)] for j in range(n)] #identity matrix with multilayer comprehension 
#lambda -> replaces the function input without a format 
print((lambda x, y: x + y)(1,2))
print(list(map(lambda x: x + 2, range(5))))
print(list(filter(lambda x: x > 4, range(10))))
#Tenary operators -> returns value1 if condition holds and if else returns value2 
value1 if condition else value2
class #creates a class -> see object_oriented_programming.py
#Module -> currently read software 
#Package -> imported software
import file #normally, the file must be in the same directory
#in python /usr/bin/python -> usr.bin.python
from directory import file #this will import file from other directory 
#if a file is imported and used in other file, __name__ == 'file' not '__main__'
if __name__ is '__main__' #this condition is used a lot to write module only codes 
import file as f #file alias is now f
from directory import * #this imports all files in the directory 
#this dot methods do not work if module read directory is the highest directory
#module reading directory enables access to all the lower files but not higher, so this must be concerned 
#use 'python -m directory' to set the module reading directory
from .file import * #one dot means current directory
from ..file import * #two dots mean one level upper directory
#if a folder is imported, __init__.py is automatically made and does importing
#basically module is importing the importing codes in __init__.py
#some standard libraries 
import time, random
#use package managing software for external packages -> currently using virtualenvwrapper and conda
#use Deque to make doubly linked list(Tree from Data Structre and Algorithms) and Queue inference(aslo from the study)
from collections import deque
queue = deque([10, 5 , 12]) #linked list
queue.appendleft(16) #left append take O(n) for python list but deque takes O(1)
queue.append(9) 
queue.pop()
queue.popleft() #also O(1) vs O(n)
deque(reversed(queue)) #reversing doubly linked list. Takes O(n)
#Use Heapq to make priority queue(from the study)
import heapq 
queue = [1,2,3,4]
heapq.heapify(queue) #making a heap 
heapq.heappush(queue, 3)
heapq.heappop(queue)
#Use Defaultdict to make dictionary data structure easier 
#useful in counting 
#normal case
text = 'hello, my name is'
characters = {}
for char in text:
    count = characters.get(char,None)
    if count is None:
        characters[char] = 0
    characters[char] += 1 
#using Defaultdict 
from collections import defaultdict
characters = defaultdict(int) #int set as default value, probably 0
for char in text:
    characters[char] += 1
#Use Counter to count
#used in dictionary's format
from collections import Counter
Counter([1,2,2,2,4]) #counts repeated integers
characters = Counter(text) 
#logic operations are possible 
a = Counter([1,1,1,2])
b = Counter([1,1,2,2])
a + b
#Use Dataclass
#decorator with a little different grammar
from dataclasses import dataclass
@dataclass
class Coords3D: #this class already exists in dataclass but manipulating using decorator is possible 
    x:float
    y:float
    z:float = 0 
    def norm(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
point = Coords3D(10,20,2)
print(point)
print(point.norm())
#Use Namedtuple 
from collections import namedtuple
#attributes can be manipulated
Coords3D = namedtuple('Coords3D', ['x','y','z'])
point = Coords3D(10,2,3)
point.x
point[1]
print(*point)
#re python package 
import re
#use regular expression for python strings
match = re.search(pattern, text, re.MULTILINE)
print(match.group(0))
#for loop is possible
for match in re.finditer(pattern, text, re.MULTILINE):
    print('whole number', match.group(0))
    print(r'\1 number only', match.group(1))
#replace all the pattern 
re.sub(pattern, replace, text, re.MULTILINE)
#splitting
re.split(pattern, text)
#compiling regular expression is much faster
compiled = re.compile(pattern, re.MULTILINE)
for string in dataset:
    match = compiled.search(string)
    print(match.group(0))
#Input and Output 
#Standard input and output -> Mac Terminal, Window CMD 
#Input through Terminal will save txt format 
python test.py > output.txt #this linux command saves output in the file 
python test.py < input.txt #this linux command puts inputs to the python program from the file 
#using both in the same time is possible 
python test1.py | test2.py #pipeline, starts test1 file then take the output of the file as the input of the next file
#also mixing these commands is possible 
#File Open 
open('file name', 'file access mode', encoding = 'encoding')
#access mode 
r #saving the data in python software as txt
rb #saving the data in python software as binary
w #creating a new file from python software as txt
wb #creating a new file from python software as binary
a #adding something to the file
file.close() #closing the access of python software to the file 
#reading the file, use read()
file = open('text.txt', 'r') #python openned the file
content = file.read() #now python software saved this file 
file.close() #now software does not access the file in the system memory
#use reserved word 'with' to omit close()
with open('text.txt', 'r') as file:
    content = file.read()
print(content)
#reading the file in lines 
#creating a python list
content = []
with open('text.txt', 'r') as file:
    for sentence in file:
        content.append(sentence)
print(content)
    content = file.readlines() #same code from the above
#this method does not omit escaping words in the txt file 
#writing a file, use write()
with open('text.txt', 'w') as new:
    for i in range(10):
        new.write(f'{i+1}th line \n')
    new.writelines(f'{i+1}th line \n' for i in range(10)) #same code from the above
#adding to a file
with open('text.txt', 'a') as original:
    original.write('adding lines')
    original.writelines(f'{i+1}th line \n' for i in range(10)) 
#Control directories 
import os 
os.mkdir('path') #making a directory(folder)
if not os.path.isdir('test'): #checking the existence of the folder
    os.mkdir('test')
os.makedirs('test/a/b/c', exist_ok = True) #creating a long directory
#exist_ok -> ignores the command if the directory already exists
#list the directories under a folder 
print(*[entry for entry in os.listdir('path')])
#use glob package to select certain files 
import glob
print(*[entry for entry in glob.glob('path/*.txt')]) #* will get any file with .txt
#use pickle package to turn python object into a txt data in the memory
import pickle
seq = [[i * j for i in range(100)] for j in range(100)] #python matrix object saved in python software 
#creating a pickle file and write in the file
with open('test.pk1', 'wb') as fd: #the object is in binary
    pickle.dump(seq, fd) #the object is in os as pickle file now
del seq #deleting the object in the python software
with open('test.pk1', 'rb') as fd:
    seq = pickle.load(fd) #the object is returned to the python software 
print(seq[1][4])
#not that used often due to security and compatibilty problem
#class object can be also turned into a txt data 
#certain types of attribute made from data structures that are not serializable will make pickling impossible 
class my_complex:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
    def __add__(self, other):
        return my_complex(
            self.real + other.real, 
            self.imaginary + other.imaginary)
my_complex = my_complex(1,2)

with open('test.pk1', 'wb') as fd:
    pickle.dump(my_complex, fd)
del my_complex #deleting class object that is saved
with open('test.pk1', 'rb') as fd:
    my_complex = pickle.load(fd)
del my_complex #deleting the class itself in the software, if the class is deleting object won't be loaded
#Csv control, comma seperated values, also, many use pandas package(pandasbasic.py)
import csv 
with open('test.csv', 'r') as fd:
    reader = csv.reader(fd, delimiter=',', 
    quotechar = '"' #text data to string,
    quoting = csv.QUOTE_MINIMAL) #parsing method: minimal
#for loop is possible after this
for entry in reader:
    print(entry)
#writing a csv
#parsing -> reinterpreting data 
with open('test.csv', 'w') as fd:
    reader = csv.writer(fd, delimiter=',', 
    quotechar = '"' #text data to string,
    quoting = csv.QUOTE_MINIMAL) #parsing method: minimal
writer.writerow(['id','label']) #single line
writer.writerows(['i', f'label_{i}' for i in range(10)]) #many lines
#JSON file useful than csv since many data structures are implemented in strings
import json
with open('test.json', 'r') as fd:
    data = json.load(fd)
print(data['hobbies']) #the string data can be managed as if they were python objects
#also writing one is possible
#but only data types: str, int, float, bool, None and structures list and dictionary are fine
#others need a decoder
obj = {
    'Id': None
    'bool': False
    'hobbies': {
        'sports': [
            'snowboard', 'bolleyball'
        ]
    }
}
with open('test.json', 'w') as fd:
    json.dump(obj, fd) #writing a json
#using secret.json file is possible
#python is porgrammed to protect the secret.json file
with open('secret.json', 'r') as fd:
    data = json.load(fd) #variable from secret.json file cannot be accessed

#XML, HTML 
#tags exist
#Beautiful Soup(webscraper project in frontend)
from bs4 import BeautifulSoup
with open('text.xml', 'r') as fd:
    soup = BeautifulSoup( #creating soup object
        fd.read(), #reading fd in python software
        'html.parser' #selecting parser
    )
to_tag = soup.find(name = 'to') #finding a tag
print(to_tag.string) #returning string in the tag
for cite_tag in soup.findAll(name='cite'): #finding all the tags
    print(cite_tag.string) #printing all the tags
cities_tag = soup.find(name = 'cities') #returning attributes under a tag
print(cities_tag.attrs) #returns all attributes
print(cities_tag['attr']) #returns the value of attribute attr only
cities_tag = soup.find(attrs= {'attr':'name'}) #returning tag from attributes is possible
for cite_tag in cites_tag.find_all(name='cite'): #returing all tag data under a found tag
    print(cite_tag.string)
#YAML, similar to JSON 
import yaml
import pprint #package that prints neatly
with open('test.yaml', 'r') as fd: #reading yaml file
    data = yaml.load(fd, Loader=yaml.FullLoader) #saved in python software 
pprint.pprint(data) #printing the data
#executing python commands
python software #in linux console 
python software arguments #argument can be selected for some version with this command 
#checking argument version 
import sys
print(sys.argv)
#for certain arguments only
python software --option 1234 
#writing these commands in python software is difficult so use this package 
import argparse
parser = arg.parse.ArgumentParser()
#short parser, long parser 
parser.add_argument('-l', '--left', type = int)
parser.add_argument('-r', '--right', type = int)
parser.add_argument('-operation', 
dest='op', #the type of the target
help='Give operation', #explanation 
default = 'sum') #default value
args = parser.parse_args()
print(args)
#controlling arguments of a python software
if args.op == 'sum':
    out = args.left + args.right 
elif args.op == 'sub':
    out = args.left - args.right
print(out)
#Excpetion Handling -> When error happens, do this 
#use 'try' and 'except' reserved words
#when the number is divided with zero, make an exception
for i in range(5,-5):
    try: 
        print(10/i)
    except ZeroDivisionError:
        print('Zero division, skip the number')
#Built-in exceptions 
IndexError #larger than defined index
NameError #non existing variable 
ZeroDivisionError #number divided with zero 
ValueError #wrong data type, ex: float('abc)
FileNotFoundError #file in os does not exist
#there is class of exception defined in python 
BaseException #SystemExit, Exception, and KeyboardInterrupt exception classes are under BaseException 
#a class of exception can be made by python users
#'raise' reserved word will act as if certain error happend in defined conditions 
try: 
    while True:
        value = input('Insert A, B, or C: ')
    if len(value) == 1, and value not in 'ABC':
        raise ValueError('Wrong Value. Exiting')
    print('Chosen option:', value)
except ValueError as e: #for this error, creating an object
    print(e) #report error through saved object
#'assert' will raise assert error when condition is false
def add_int(parameter):
    assert isinstance(parameter, int), 'int only' #error message can be added after the condition 
    return parameter + 1
try:
    print(add_int(10))
    print(add_int('str'))
except AssertionError as e:
    print(e)
#Post error processing 
try: 
    function()
except SomeError as e:
    print(e, 'Exception occurred')
print('Exception passed') #this will be printed no matter what 
#for this code, when other error not in SomeError occurs, the software will end 
try:
    function():
except SomeError as e:
    print(e, 'Error occurred')
else: 
    print('No Error') 
#with else loop, this will print when there is no error only
try: 
    function():
except SomeError as e:
    print(e, 'Exception occurred')
finally: 
    print('Exception passed') #with finally loop, the message will be printed even other error occurs
#Logging 
#saving the changes and interactions in using the software 
#there are many purpose for logging: user pattern saving, debugging, etc.. 
import logging 
#defining logging circumstances 
#python default levels
logging.debug('for debugging')
logging.info('user information')
logging.warning('might give problems')
logging.error('error occurs')
logging.critical('critical error')
#python will save different information for these levels
#root logging -> default
logging.basicConfig(
    filename='test.log' #logging will be saved here
    level = logging.INFO
)
logging.debug('this won\'t be written')
logging.info('this will be written')
logging.error('this as well')
#new levels and logger can be defined 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) #this logger will save Debugging info as well 
#so the software will have two loggers: root(INFO level) and new logger(DEBUG level)
#Web
import requests
#user computer sends request to company server and the server sends html page
#the browser implements the html 
#this can be done in python as well with 'requests' library 
url = 'https://9gag.com'
response = requests.get(url)
print(response.status_code) #shows the connection status
print(response.text) #returns html code 
#crawling
from bs4 import BeautifulSoup
url = 'https://sports.news.naver.com/index'
response = requests.get(url)
#creating a soup object
soup = BeautifulSoup(
    response.text,
    'html.parser'
)
headline = soup.find(name='ul', attrs = {'class':'today_list'})
for title in headline.find_all(name='strong', attrs = {'class':'title'})
#use 'scrapy' for interactive web
#network structure  