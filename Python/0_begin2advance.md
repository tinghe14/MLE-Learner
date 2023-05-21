Course Contents
1. Introduction
2. [The Absolute Bare Minimum](#section_2): varaibles and data types, type conversion, basic operator, lists, stinrgs, statements and indentation, assignment statements, if statements, while and for loops, print and import statements, input from users, functions, coding styoe guide.
4. [Better Tools](#section_3): objects, lists, tuples, sets, dictionaries, string methods, loops for objects, [handling exceptions](#exceptions), type conversions, scope, file i/o, pickling
5. Classes
6. Getting Fancy
7. Testing
8. Graphical User Interface
9. Modules
10. Appendix
11. Exam

# The Absolute Bare Minimum
<a id='section_2'></a>
### Variables and Data Types
variables don't need to be decleared. Rather, varaibles come into existence when you assign a value to them
important data types:
- integers
- floats(reals)
- booleans(logicals)
- strings
- lists: can store multiple items of different data types inside of them. These items are stored with a fixed and defined order and are easily changeable
- NoneType: only has a single value, None, this is a actual value that can be assigned to variables
    - Note: None is not a default value for values that haven't yet been assigned a value. If users did try to read an unassigned variables, this will result in an error
- tuples
- sets
- dictionaries
### Type Conversion
conversion function: float(), int(), str()
check data type of variables: type()
round(): can be used to convert floating-point numbers to nearest interger
### Basic Operators
operand_a->operator->operand_b=result
arithmetic operators: 
- Except in the case of division (which always yields a float), the result of an operation will be an integer if both the operands are integers. Otherwise, it will be a float.
- addition, substraction, multiplication, exponentiation, division(/), interger divide(//, round down), modulus(%, remainder of division)
- order of precedence:
    - from highest precedence to the lowest one: (), **, * and /, + and -
    - multiple expoentiations are done $`\textcolor{red}{\text{right to left}}`$
    ~~~
    expr = 2 ** 3 ** 4 # Original expression
    expr_right = 2 ** (3 ** 4) # Right-sided expression calculated first
    expr_left = (2 ** 3) ** 4 # Left-sided expression calculated first
    # Original:  2417851639229258349412352
    # Right Sided:  2417851639229258349412352
    # Left Sided:  4096
    ~~~
comparison operators:
- operator: ==, !=, <,<=, >,>=
- $`\textcolor{red}{\text{chaining语法糖}}`$: unlike most language, python lets chain comparisons
~~~
print(2 < 3 < 4 > 2) # Chaining Of operators (will check pairwise) -> True
print(2 > 3 < 4 > 2) # Chaining Of operators (will check pairwise) -> False
~~~
logical operators:
- operator: and (True if both statements True), or (True if either one True), not
- and and or operators are $`\textcolor{red}{\text{short-circuit operators语法糖}}`$: the second operand is evaluated only if necessary
- other important points when dealing with logical operators:
    - in a numeric experession, True has the value 1 and False has the value 0
    - in a boolean expression, all numeric zero values are False, and all nonzeor values are True
    - the special None value indicates 'no value', and it is treated as False
### Lists
a sequence of values enclosed in squared brackets. can store multiple items of different data types inside of it
### Strings
a sequence of characters enclosed in quotation marks
- concatenation: + operator
- special characters: escaping character(转义字符用backslash\表示):\n newline, \t tab, \" double quote, \' single quote, \\ backslash, \Uxxxx unicode
- indexing: access individual characters of strings
### Statements and Indentation
standard indentation is four spaces
- if you want to put two statements on the same line, can separate them with a semicolon, ;
### Assignment Statements
- assignment shortcuts: an operator and a single = sign,+=
- walrus operator: allows you to assign a value to a variable and return its value within an expression,:=
~~~
n = 10
x = (n := 3 * n) + 1 # Using walrus operator
print(n) # give 30
print(x) # give 31
~~~
### If Statements
### While and For loops
while loops continue to execute as long as the end condition is True.
- The walrus operator can also be used in while loops
~~~
n = 5
while n > 0: # Will keep on running till n is greater than 0
  print(n)
  n -= 1 # Without this, it will get stuck in infinite loop
# same as 
n = 6
while (n := n - 1) > 0: # Decrementing and then checking if greater than 0 together
  print(n)
~~~
For loops are used for iterating over each element of sequence data types: strings, lists, tuples, etc.
- range() function: range(a, b, c) will give the numbers a, a+c, a+c+c, a+c+c+c, etc., up to (or down to), but not including, b
    - the value returned by a range() function is not a list, but rather an iterator, if need a list, use list(range(...))
### Print and Import Statement
not built in function can  be imported from other module, a module is just a file containing code
### Input from the User
- ask the user for input: input(_promopt_): If you omit the prompt, the user will be left staring at a blank screen and wondering why the program isn’t doing anything.
~~~
name = input("What is your name? ")
print("Hello,", name)
# the result of a call to input is always string. 
#if you are expecting an integer or float, can use int or float functions
age = int(input("How old are you, " + name + "? "))
# if the user types in something can't be made into integer, this will result an error
# error handling will be introduce later
~~~
### Fuctions
Functions are a series of statements that are used to perform a specific operation.
two types of functions: built-in functions and custom-defined functions
syntax of function:
- def keyword along with the name of the function, a parehthesized list of parameters, and a colon. This is followed by the indented body of function
- return keyword is used for stopping the execution of the function statement and allows you to return a value from the function
- triple-quote string: to document the purpose of the function; documenting each function is optional but strongly recommend. 
    - for any function with a documentation string. help(function_name) will print that string
- every function returns a value
    - if you don't specify a return value, the function will return None (This is an actual legal value of type NoneType, so you can assign it to a variable or ask if a variable is equal to it.)
        
$`\textcolor{red}{\text{Local Function}}`$ :
- a function may be defined within another function, and it becomes local to the function in which it has been defined
### Coding Style Guide and Conventions惯例
- Evaluating a function definition (def) causes the function to be defined, but it won’t be executed until some other statement calls it.
- It is common for a program to consist of a collection of functions, with the last line of the program being a single top-level call to a “main” function
### Test
1. write a function checks whether the given number is a prime number or not
- prime numbers are numbers that are only divisible by itself and 1
- negative numbers, which are also not prime numbers
~~~
def is_prime(n):
    """Tests if a number n is prime."""
    if n <= 1: # Adding this condition
        return False
    divisor = 2
    while divisor <= n // divisor: #检测一半的n就行 换算理解divisor**2 <= n
        if n % divisor == 0:
            return False
        divisor += 1
    return True
~~~
2. implementing the Fibonacci Series
- Fibonacci sequence is a series of numbers where every number is the sum of the two numbers before it. The first two numbers are 0 and 1
- instead of using recursion, your function must use loops
~~~
def fib(n):
    first, second = 0, 1
    if n < 1: return -1
    if n == 1: return first 
    if n == 2: return second
    count = 3
    while count <= n:
        fib_n = first + second
        first = second 
        second = fib_n
        count += 1
    return fib_n
~~~
3. check whether the brackets are balanced
- Given a string containing only square brackets, [], you must check whether the brackets are balanced or not. The brackets are said to be balanced if there is a closing bracket for every opening bracket 
~~~
def check_balance(brackets):
    check = 0
    for bracket in brackets:
        if bracket == '[':
            check += 1
        elif bracket == ']':
            check -= 1
        if check < 0:
            break
    return check == 0
~~~

# Better Tools
<a id='section_3'></a>
### Objects
