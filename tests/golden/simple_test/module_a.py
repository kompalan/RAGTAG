import module_b

def function_a():
    print("hello!")
    module_b.function_c()

class A:
    def function_d(self):
        function_a()
