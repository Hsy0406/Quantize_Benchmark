import measure as mea
import new_rules as rules
import tvm
import tvm.topi.testing
import tvm.topi.cuda
import tvm.relay.quantize
import numpy as np
import tvm.autotvm
from tvm.contrib import graph_runtime
import transformer_test
import copy
import heapq
import random
import xlsxwriter as xw



def check_tensorcore(state):
    dtype_4 = False
    dtype_8 = False   
    for dtype in state.dtypes:
        if(dtype == 'int4'):
            dtype_4 = True
        elif(dtype == 'int8'):
            dtype_8 = True
    if(dtype_8 and dtype_4):
        return "tensorcore4_8"
    elif(dtype_8 or dtype_4):
        return "no_tensorcore"
    else:
        return "no_tensorcore"
    

class state_data:
    def __init__(self, state, time) -> None:
        self.state = state
        self.time = time
    def __lt__(self, data):
        return self.time < data.time
    
class SearchPolicy:
    def __init__(self, data_write = [], file = 'out.txt'):
        self.measured_states_set_ = set()
        self.file = file
        self.data_write = data_write

    def evaluate(self, state, cur_layer, file):
        # print(state.ToStr())
        if(check_tensorcore(state) == "tensorcore"):
            # try:
            ftimer_time, timeuse = transformer_test.measure_t(state.batch, state.M, state.N, state.K, state.points_x, state.points_y, 
                                    state.shapes_x, state.shapes_y, state.dtypes, state.scales, cur_layer)
                
            # except:
            #     ftimer_time = float("inf")
            #     timeuse = float("inf")
        elif(check_tensorcore(state) == "no_tensorcore"):  
            # try:
            ftimer_time, timeuse = transformer_test.measure(state.batch, state.M, state.N, state.K, state.points_x, state.points_y, 
                                    state.shapes_x, state.shapes_y, state.dtypes, state.scales, cur_layer)
            # except:
            #     ftimer_time = float("inf")
            #     timeuse = float("inf")
        else:
            # try:
            ftimer_time, timeuse = transformer_test.measure_t48(state.batch, state.M, state.N, state.K, state.points_x, state.points_y, 
                                    state.shapes_x, state.shapes_y, state.dtypes, state.scales, cur_layer)
            # except:
            #     ftimer_time = float("inf")
            #     timeuse = float("inf")
        d = {'s': 32, 'n': len(state.shapes_y), 't': timeuse}
        self.data_write.append(d)
        file.write(state.ToStr()+str(timeuse))
        file.write('\n')
        return timeuse #越小越好


    
    def Search(self, init_state, n_trails = 100, encoder_layer = 12):
        cur_layer = 0
        ct = 0
        print("Search: start",cur_layer,"th layer searching")
        print("Search: start",ct,"th times")
        time = self.evaluate(init_state, cur_layer, self.file)
        init_bs = state_data(init_state, time)
        best_state = [init_bs]
        now_state = copy.deepcopy(init_state)
        while(cur_layer < encoder_layer): 
            unchange = 0
            cur_layer += 1
            print("Search: start",cur_layer,"th layer searching")
            while(ct < n_trails):
                ct += 1
                print("Search: start",ct,"th times")
                fusek = rules.kFuseRule()
                fusen = rules.nFuseRule()
                sortn = rules.SortRule()   #先对n轴进行排序，从小到大
                sortk = rules.SortRule(2)   #再对k轴进行排序，从小到大
                if(sortn.meetcondition(now_state)):
                    now_state = sortn.apply(now_state)
                if(fusek.meetcondition(now_state)):
                    new_state = fusek.apply(now_state)
                    if(new_state.ToStr() not in self.measured_states_set_):
                        time = self.evaluate(new_state, cur_layer, self.file)
                        bs = state_data(new_state, time)
                        heapq.heappush(best_state, bs)
                        self.measured_states_set_.add(new_state.ToStr())
                        now_state = copy.deepcopy(new_state)
                        unchange = 0
                        continue
                if(sortk.meetcondition(now_state)):
                    now_state = sortk.apply(now_state)
                if(fusen.meetcondition(now_state)):
                    new_state = fusen.apply(now_state)
                    if(new_state.ToStr() not in self.measured_states_set_):
                        time = self.evaluate(new_state, cur_layer, self.file)
                        bs = state_data(new_state, time)
                        heapq.heappush(best_state, bs)
                        self.measured_states_set_.add(new_state.ToStr())
                        now_state = copy.deepcopy(new_state)
                        unchange = 0
                        continue
                unchange += 1
                
                if(unchange == 30):
                    break
        return best_state[0]
    

def random_choice():
    data_list = ['int8', 'int8', 'int16', 'int32']
    num = random.choice([0, 1, 2, 3])
    return data_list[num]

def data_scale(data_type):
    if(data_type == 'int4'):
        return 8.0/2**3
    elif(data_type == 'int8'):
        return 8.0/2**7
    elif(data_type == 'int16'):
        return 8.0/2**15
    else:
        return 8.0/2**31
    

def xw_toExcel(data, fileName):
    workbook = xw.Workbook(fileName)
    worksheet1 = workbook.add_worksheet("sheet1")
    worksheet1.activate()
    title = ['size', 'numbers', 'time']
    worksheet1.write_row('A1', title)
    i = 2
    for j in range(len(data)):
        insertData = [data[j]["s"], data[j]["n"], data[j]["t"]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()
    
if __name__ == "__main__":
    with open("output.txt", "w") as file:
        d_model = 768   # hidden size
        sequence_len = 256 # token length
        batch_size = 2
        encoder_input_shape = (batch_size, sequence_len, d_model)
        n_trails = 100
        data_write = []
        encoder_layer = 1

        point_x = []
        point_y = []
        shape_x = []
        shape_y = []
        dtypes = []
        scales = []
        granularity = 16  # quantized pattern granularity
        # genrate the quantized pattern based on the granularity
        for i in range(int(d_model/granularity)):    # the range = d_model (i.e. hidden size) / pattern_size (i.e. 32)
            for j in range(int(d_model/granularity)):
                point_x.append([0, granularity*j])
                point_y.append([granularity*i, granularity*j])
                shape_x.append([2, sequence_len, granularity])
                shape_y.append([2, granularity, granularity])
                data = random_choice()
                dtypes.append(data)
                scales.append(data_scale(data))

        searchpolicy = SearchPolicy(data_write, file)
        init_state = mea.State(batch_size, sequence_len, d_model, d_model, point_x, point_y, shape_x, shape_y, dtypes, scales, "float32")
        result = searchpolicy.Search(init_state, n_trails, encoder_layer)
        xw_toExcel(data_write, str(granularity)+'.xlsx')
        # print(result.state.ToStr())
        print(result.time)



