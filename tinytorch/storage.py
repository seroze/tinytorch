import array 

class Storage:

    def __init__(self, size, dtype = 'd'):
        self.size = size 
        self._data = array.array(dtype, [0.0]*size) # 'd' = double ('float')

    def __getitem__(self, index):
        return self._data[index] 

    def __setitem__(self, index, value):
        self._data[index] = value 

    def __len__(self):
        return self.size