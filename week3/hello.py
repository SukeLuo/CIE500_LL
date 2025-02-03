print('Hello World')
print('This is Linyue')
print('haha')

class car:
    def __init__(self, color, model):
        self.color = color
        self.model = model
    
    def mymodel(self):
        print(f'car model is {self.model}')
    
if __name__ == "__main__":
    var1 = 0
    a = car('black', 'mini')
    b = car('white', 'nissan')
    a.mymodel()
    print(a.color, b.model)