from module_b import function_e, B


class A(B):
    def function_d(self):
        super().function_c()


def function_a():
    print("hello!")
    a = A()
    a.function_d()
    function_e()


function_a()
