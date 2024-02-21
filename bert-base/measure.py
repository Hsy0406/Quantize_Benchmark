import tvm
from scipy import spatial


class State:
    def __init__(self, batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, input_data):
        self.batch = batch
        self.M = M
        self.N = N
        self.K = K
        self.points_x = points_x
        self.points_y = points_y
        self.shapes_x = shapes_x
        self.shapes_y = shapes_y
        self.dtypes = dtypes
        self.scales = scales
        self.input_data = input_data
        self.max_tile_nums = 20
        
    def check_valid(self):
        for i in range(0, len(self.points_y)):
            if(self.points_x[i][0] < 0 or self.points_x[i][1] < 0 or self.points_y[i][0] < 0 or self.points_y[i][1] < 0 or self.shapes_x[i][1] < 0 or
               self.shapes_x[i][2] < 0 or self.shapes_y[i][1] < 0 or self.shapes_y[i][2] < 0):
                return False
        return True
        
    def ToStr(self):
        state_str = str(self.M) + str(self.N) + str(self.K) + str(self.points_x) + str(self.points_y) + str(self.shapes_x) + str(self.shapes_y) + str(self.dtypes)
        return state_str
    
    
class Measure:
     def __init__(self):
         self.measureresult = []
         self.best_ct = 0
         self.time = float("inf")
         

class MeasureResult:
    def __init__(self, state, time, state_str):
        self.state = state
        self.time = time
        self.state_str = state_str
    
