class A:
    def __init__(self) -> None:
        self.gen = (i for i in range(1000))

    def Output(self):
        yield next(self.gen)
    
class B:
    def __init__(self) -> None:
        self.data = []
    
    def Input(self, info):
        self.data.append(next(info))
        print(self.data)

a = A()
amethod = a.Output
b = B()
bmethod = b.Input

bmethod(amethod())
bmethod(amethod())