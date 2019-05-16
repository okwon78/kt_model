

a = 11

def func():


    def f():
        global a
        if a > 10:
            a = 1


    f()




func()

