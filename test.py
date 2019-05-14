# def fun(fun2):
#     # print('hahaha')
#     def fun1():
#         print("lalala")
#         fun2()
#     return fun1

# @fun
# def fun2():
#     print('gagaga')


# fun2()

def fun(*kwargs):
    print(kwargs)

fun(1,2,3,4)

def add(a, b):
    return a+b
def add(a, b, c):
    return a + b + c
print(add(1,2))
print(add(1,2,3))