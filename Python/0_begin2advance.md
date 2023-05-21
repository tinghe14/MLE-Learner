Course Contents
1. Introduction
2. [The Absolute Bare Minimum](#section_2): varaibles and data types, type conversion, basic operator, lists, stinrgs, statements and indentation, assignment statements, if statements, while and for loops, print and import statements, input from users, functions, coding styoe guide.
4. Better Tools
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
