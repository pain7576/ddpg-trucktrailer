
class fifo:
    def __init__(self, maxsize):
        self.maxsize = maxsize

    def add_to_fifo(self, name , item):
       name.append(item)
       if len(arr) > self.maxsize:
           arr.pop(0)

# Usage
fifo = fifo(3)
arr = []


fifo.add_to_fifo(arr, 1)
fifo.add_to_fifo(arr, 2)
fifo.add_to_fifo(arr, 3)
fifo.add_to_fifo(arr, 4)
fifo.add_to_fifo(arr, 5)

print(arr)