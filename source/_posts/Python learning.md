---
title: Tutorial for PyCharm
top: false
cover: false
toc: true
mathjax: true
date: 2022-04-06 22:58
password:
summary:
tags:
- python
categories:
- programming
---

## Tutorial for PyCharm

`Alt` `Enter` preview the warning and apply a quick fix.

`Ctrl` `Shift` `A` is the shortcut of Find Action. Alternatively search Everywhere can be invoked by pressing `Shift` twice.

`Ctrl` `W`, once to select the word under the caret, twice to select the whole string, tree times to include quotes, four times to select the whole call. If the caret is at the beginning of a statement, twice pressing is used to select it 

`Ctrl` `Shift` `W`, to shrink selection back to the arguments. 

`Ctrl` `/`， to comment the code. Also `#` before code or `“”“ ”“”` or`‘'' '''` between the block

`ctrl` `P` to give the values a method expects

## For beginner

### Variables

Boolean data type is using “True” and “False”. Python is sensitive with lowercase and uppercase letters

String concatenation: 

```python
print(“Text ” + data_name)
```

Input function: 

```python
name = input (“What is your name”)
```

When you want to name a variable, it is not required to add the type of it. So, if we want to convert the variable from one type to another, we can use:

```python
int(name)
float()
bool()
str()
```

### Operators

Comparison operators:

```python
>
<
<=
>=
==
!=
```

logical operators:

```python
print(price>1 and price <2)
print(price>1 or price<0)
print(not price>1)
```

a Boolean value will be output

### Lists

```python
names = ['John', 'Mary', 'Bob', 'Jack']
print(names[0:2]) #output will be John Mary
print(names[-1]) #output will be Jack
```

.append(): add element into the lists

.insert(): insert an element

.remove(): remove an element

.clear(): empty a list

len(): give the numbers of elements

```
for items in numbers:
print(items)
```

Each items can automatically hold one elements

### Range

range(a, b , c): give a sequence of numbers. a for beginning number, b for ending number and c for step. a, c can be default to be 0 and 1

### Tuples

```
numbers =(1, 2, 3, 4)
```

it is immutable, we can not change elements

# Tricks 

## and 和 or 的短路效应：

当or表达式里的所有值为真，会选择第一个值

当and表达式里所有值为真，会选择第二个值

## intern 机制

intern（字符串驻留）的机制在Python解释器中被使用，

当有空格，或者字符串长度超过20个字符，则不启动intern机制

```Python
s1="hello"
s2="hello"
s1 is s2 # True

s1="hell o"
s2="hell o"
s1 is s2 # False

s1="hello"*4
s2="hello"*4
s1 is s2 # False

s1="hello"*5
s2="hello"*5
s1 is s2 # True
```

## argument 和 parameter 的区别

parameter：形参（formal parameter），体现在函数内部，作用域是这个函数体。
argument ：实参（actual parameter），调用函数实际传递的参数。

## return不一定都是函数的终点

在try…finally…语句中，try中的 return 会被直接忽视（这里的 return 不是函数的终点），因为要保证 finally 能够执行。

```python
def func():
	try:
		return 'try'
	finally:
		return 'finally'
func() #'finally'

def func1():
	try:
		return 'try'
	finally:
		print('finally')
func1() 
#finally
#'try'
```

如果 finally 里有显式的 return，那么这个 return 会直接覆盖 try 里的 return，而如果 finally 里没有 显式的 return，那么 try 里的 return 仍然有效。
