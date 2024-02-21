import mix_batch_matmul_relay_test as m
import measure as mea
import cost_model
import rules
import tvm
import os
from tvm import te, relay, autotvm, auto_scheduler
import tvm.topi.testing
import tvm.topi.cuda
import tvm.relay.quantize
import numpy as np
import tvm.autotvm
from timeit import default_timer as timer
from tvm.contrib import graph_runtime
import random
import copy
import heapq

params={
             "eps_greedy": 0.4,
            "sample_init_min_population": 10,
            "sample_init_use_measured_ratio": 0.4,
            "evolutionary_search_num_iters": 4,
            "retry_search_one_round_on_empty": 1,
            "evolutionary_search_population": 512,
            "num_measures_per_round":32,
            "evolutionary_search_mutation_prob": 0.85,
            "num_measure_per_iter": 100
        }

def ComputePrefixSumProb(weights, prefix_sum_probs):
    sum = 0.0
    for i in range(0, len(weights)):
        sum += max(weights[i], 0.0)
        prefix_sum_probs[i] = sum
    for i in range(0, len(weights)):
        prefix_sum_probs[i] = prefix_sum_probs[i]/sum
        
    
def RandomChoose(prefix_sum_probs):
    x = np.random.uniform(0, 1)
    for index, prob in enumerate(prefix_sum_probs):
        if x <= prob:
            return index
    return len(prefix_sum_probs)-1 

def random_choice_index(lst):
    index = np.random.randint(0, len(lst))
    return index

def RandomSampleStates(in_states, out_size):
    out_states = []
    for _ in range(out_size):
        out_states.append(np.random.choice(in_states))
    return out_states

def Argsort(scores):
    index = list(range(len(scores)))
    index.sort(key=lambda x: scores[x], reverse=True)
    return index

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
        return "tensorcore"
    else:
        return "no_tensorcore"
    

class SearchPolicy:
    def __init__(self, program_cost_model, init_rules):
        self.program_cost_model = program_cost_model
        self.init_rules = init_rules
        self.measured_states_set_ = set()
        self.measured_states_vector_ = []
        self.measured_states_throughputs_ = []

    def evaluate(self, state, file):
        if(check_tensorcore(state) == "tensorcore"):
            # try:
            ftimer_time, timeuse = m.measure_t(state.batch, state.M, state.N, state.K, state.points_x, state.points_y, 
                                    state.shapes_x, state.shapes_y, state.dtypes, state.scales)
                
            # except:
            #     ftimer_time = float("inf")
            #     timeuse = float("inf")
        elif(check_tensorcore(state) == "no_tensorcore"):  
            # try:
            ftimer_time, timeuse = m.measure(state.batch, state.M, state.N, state.K, state.points_x, state.points_y, 
                                    state.shapes_x, state.shapes_y, state.dtypes, state.scales)
            # except:
            #     ftimer_time = float("inf")
            #     timeuse = float("inf")
        else:
            # try:
            ftimer_time, timeuse = m.measure_t48(state.batch, state.M, state.N, state.K, state.points_x, state.points_y, 
                                    state.shapes_x, state.shapes_y, state.dtypes, state.scales)
            # except:
            #     ftimer_time = float("inf")
            #     timeuse = float("inf")
        file.write(state.ToStr()+str(ftimer_time))
        file.write('\n')
        return ftimer_time #越小越好


    def PickStatesWithEpsGreedy(self, best_states, random_states, remaining_n_trials):
        num_random = params["eps_greedy"]*params["num_measures_per_round"]
        num_good = params["num_measures_per_round"] - int(num_random)
        inputs = []
        offset_best = 0
        offset_random = 0
        
        while (len(inputs) < min(params["num_measures_per_round"], remaining_n_trials)):
            has_best = offset_best < len(best_states)
            has_random = offset_random < len(random_states)
            if (len(inputs) < num_good):
                if (has_best):
                    state = best_states[offset_best]
                    offset_best += 1
                elif (has_random):
                    state = random_states[offset_random]
                    offset_random += 1
                else:
                    break
            else:
                if (has_random):
                    state = random_states[offset_random]
                    offset_random += 1
                elif (has_best):
                    state = best_states[offset_best]
                    offset_best += 1
                else:
                    break
            state_str = state.ToStr()
            if (state_str not in self.measured_states_set_):
                self.measured_states_set_.add(state_str)
                inputs.append(copy.deepcopy(state))
                
        return inputs  # state列表
    
    
    def EvolutionarySearch(self, init_population, out_size):
        best_states = []
        population = params["evolutionary_search_population"]   
        mutation_prob = params["evolutionary_search_mutation_prob"]
        num_iters = params["evolutionary_search_num_iters"]
        
        if(self.program_cost_model == cost_model.RandomCostModel):
            num_iters = 2
        
        states_buf1 = copy.deepcopy(init_population)
        states_buf2 = []
        
    ## A heap to keep the best states during evolution
        heap = []
        in_heap = self.measured_states_set_
        
        max_score = -1e-10
        mutation_success_ct = 0
        mutation_fail_ct = 0
        
        class  my_data:
            def __init__(self, state, scores) -> None:
                self.state = state
                self.scores = scores
            def __lt__(self, data):
                return self.scores > data.scores
        
        for k in range(0, num_iters + 1):
            ## Maintain the heap
            pop_scores = []
            pop_scores = self.program_cost_model.Predict(states_buf1)
            pop_selection_probs = ["" for _ in range(len(pop_scores))]
            for i in range(0, len(states_buf1)):
                state = states_buf1[i]
                state_str = state.ToStr()
                if (state_str not in in_heap):
                    if (len(heap) < out_size):
                        heapq.heappush(heap, my_data(state, pop_scores[i]))
                        in_heap.add(state_str)
                    elif (pop_scores[i] > heap[0].scores):
                        old_state_str = heap[0].state.ToStr()
                        if(old_state_str in in_heap):
                            in_heap.remove(old_state_str)
                            in_heap.add(state_str)
                            heapq.heappop(heap)
                            heapq.heappush(heap, my_data(state, pop_scores[i]))
                        else:
                            continue
                    if (pop_scores[i] > max_score):
                        max_score = pop_scores[i]
        
            ## Print statistical information
            if (k % 5 == 0 or k == num_iters):
                print("GA Iter: ", k)
                if (heap != []):
                    print("\tMax score: ", max_score)
                    print("\tMin score: ", heap[0].scores)
                else:
                    print("\tMax score: N/A\tMin score: N/A")
                print("\t#Pop: ", len(heap))
                print("\t#M+: ", mutation_success_ct / (k + 1))
                print("\t#M-: ", mutation_fail_ct / (k + 1))
            if (k == num_iters):
                break
            
            ComputePrefixSumProb(pop_scores, pop_selection_probs)
            while (len(states_buf2) < population):
                ## Do mutation
                tmp_state = states_buf1[RandomChoose(pop_selection_probs)]
                rand = np.random.uniform(0, 1)
                if(rand < mutation_prob):
                    ran_num = np.random.uniform(0, 1)
                    if(ran_num < 0.5 and len(tmp_state.points_y) > 1):
                        success = False
                        index1, index2 = random.sample(range(0, len(tmp_state.points_y)), 2)
                        for fuse_rule in [rules.FuseAndFillRule(), rules.FuseOverlapRule(), rules.FusePointRule()]: 
                            if(fuse_rule.meetcondition(tmp_state, index1, index2)):
                                new_state = fuse_rule.apply(tmp_state)
                                mutation_success_ct = mutation_success_ct + 1
                                success = True
                                states_buf2.append(new_state)
                                break
                        if(not success):
                            mutation_fail_ct = mutation_fail_ct + 1
                    else:
                        choice_rule = rules.DataRule()
                        index = random_choice_index(tmp_state.dtypes)
                        if(choice_rule.meetcondition(tmp_state, index)):
                            new_state = choice_rule.apply(tmp_state)
                            mutation_success_ct = mutation_success_ct + 1
                            states_buf2.append(new_state)
                        else:
                            mutation_fail_ct = mutation_fail_ct + 1
                else:
                    states_buf2.append(tmp_state)
            states_buf1 = copy.deepcopy(states_buf2)
            states_buf2 = []            
        for item in heap:
            best_states.append(item.state)
            
        print("EvolutionarySearch\t\t#s: ", len(best_states))
        return best_states
            
        
        
    def SampleInitPopulation(self, state):
        out_states = []
        assert len(state.points_y) <= state.max_tile_nums
        population = params["evolutionary_search_population"]
        fail_ct = 0
        explored_state_strs = set()
        unchange_cnt = 0
        cand_states = [copy.deepcopy(state)]
        while(len(out_states) < params["sample_init_min_population"]):
            rand_index = random_choice_index(state.dtypes)
            data_rule = rules.DataRule()
            if(data_rule.meetcondition(state, rand_index)):
                state_new = data_rule.apply(state)
                cand_states.append(state_new)
            else:
                fail_ct += 1
            
            unchange_cnt += 1
            
            for i in range(0, len(cand_states)):
                state_str = cand_states[i].ToStr()
                if(state_str not in explored_state_strs):
                    explored_state_strs.add(state_str)
                    out_states.append(cand_states[i])

                    unchange_cnt = 0
                else:
                    fail_ct += 1
            if(unchange_cnt > 32):
                break
        return out_states
        


    
    def Search(self, n_trials, early_stopping, measure, init_state, file):
        if n_trials <= 1:
            best_states = self.SearchOneRound(0, init_state)
            assert best_states.size() > 0
            return best_states[0]
        else:
            num_random = params["eps_greedy"]*params["num_measure_per_iter"]
            ct = 0
            empty_retry_count = params["retry_search_one_round_on_empty"]
            best_states = []
            random_states = []
            inputs = []
            while (ct < n_trials):
                print("Search")
                best_states = self.SearchOneRound(num_random * 3, init_state, random_states)
                inputs = self.PickStatesWithEpsGreedy(best_states, random_states, n_trials - ct) # 选择某些state进行性能测量
                if (inputs == []):
                    empty_retry_count -= 1
                    if (empty_retry_count > 0):
                        continue
                    else:
                        print("It seems all candidates in the search space have been measured.")
                        break
                else:
                    empty_retry_count = params["retry_search_one_round_on_empty"]
                print("Measure")
                for input in inputs:
                    if(input.check_valid()):
                        input_time = self.evaluate(input, file)
                        m = mea.MeasureResult(input, input_time, input.ToStr())
                        if measure.time > input_time:
                            measure.time = input_time
                            measure.best_ct = ct
                            measure.measureresult.append(m)
                        self.measured_states_throughputs_.append(input_time)
                        self.measured_states_vector_.append(input)

                ct += len(inputs)
                if (ct - measure.best_ct > early_stopping):
                    print("Stop early since no performance improvement in the last ",early_stopping)
                    break
            
            print("Done")
            if(measure.measureresult == []):
                input_time = self.evaluate(init_state, file)
                m = mea.MeasureResult(init_state, input_time, init_state.ToStr())
                return m
            else:
                return measure.measureresult[len(measure.measureresult) - 1]
    
    
    def SearchOneRound(self, num_random_states, init_state, random_states = []):
        population = params["evolutionary_search_population"]
        num_use_measured = min(len(self.measured_states_vector_), int(params["sample_init_use_measured_ratio"]*population))
        init_population = self.SampleInitPopulation(init_state)
        indices = Argsort(self.measured_states_throughputs_)
        for i in range(0, num_use_measured):
            init_population.append(self.measured_states_vector_[indices[i]])
        if (num_random_states > 0 and random_states != []):
            random_states = RandomSampleStates(init_population, num_random_states)
        return self.EvolutionarySearch(init_population, params["num_measure_per_iter"] * 2)

               
        

if __name__ == "__main__":
    with open("output.txt", "w") as file:
        n_trails = 300
        early_stopping = 100
        point_x = [[0, 0], [0, 64], [0, 128]]
        point_y = [[0, 0], [0, 64], [0, 128]]
        shape_x = [[16, 512, 64], [16, 512, 64], [16, 512, 64]]
        shape_y = [[16, 128, 64], [16, 128, 64], [16, 256, 64]]
        dtypes = ['int16', 'int32','int32']
        scales = [8.0/2**16, 8.0/2**32, 8.0/2**32] 
        measure = mea.Measure()
        searchpolicy = SearchPolicy(cost_model.RandomCostModel(), rules.DataRule())
        init_state = mea.State(16, 512, 512, 512, point_x, point_y, shape_x, shape_y, dtypes, scales, "float32")
        state = mea.State(16, 512, 512, 512, [[0, 64], [0, 128], [0, 0]], 
                    [[32, 64], [32, 128], [0, 0]], [[16, 512, 64], [16, 512, 128], [16, 512, 128]], [[16, 128, 64], [16, 128, 128], [16, 128, 128]], 
                    ["int4", 'int8', 'int16'], [8.0/16, 8.0/256, 8.0/2**16], "float32")
        # res = m.measure(16, 512, 512, 512, point_x, point_y, shape_x, shape_y, dtypes, scales)
        # print(res)
        result = searchpolicy.Search(n_trails, early_stopping, measure, state, file)
        print(result.state_str)
        print(result.time)