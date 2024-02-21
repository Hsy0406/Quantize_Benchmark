import tvm
import random
import measure
import copy

class SearchRules:
    def __init__(self, rule_name):
        self.rule_name = rule_name

    def GetDtypeNum(self, state, index):
        dtype = state.dtypes[index]
        if(dtype == 'int4'):
            return 4
        elif(dtype == 'int8'):
            return 8
        elif(dtype == 'int16'):
            return 16
        else:
            return 32

    def GetDtype(self, num):
        if(num == 4):
            return 'int4'
        elif(num == 8):
            return 'int8'
        elif(num == 16):
            return 'int16'
        else:
            return 'int32'
        
    def GetScale(self, dtype):
        if(dtype == 'int4'):
            return 8.0/2**4
        elif(dtype == 'int8'):
            return 8.0/2**8
        elif(dtype == 'int16'):
            return 8.0/2**16
        else:
            return 8.0/2**32



class SortRule(SearchRules):
    def __init__(self, axis = 1, rule_name = "sort_rule"):
        super().__init__(rule_name)
        self.axis = axis

    def meetcondition(self, state):
        if(len(state.shapes_y) < 2):
            return False
        for i in range(1, len(state.shapes_y)):
            if(state.shapes_y[i][self.axis] < state.shapes_y[i - 1][self.axis]):
                return True
        return False
    
    def apply(self, state):
        points_x = []
        points_y = []
        shapes_x = []
        shapes_y = []
        dtypes = []
        scales = []
        sorted_indices = [idx for idx, _ in sorted(enumerate(state.shapes_y), key=lambda x: x[1][self.axis])]
        for index in sorted_indices:
            points_x.append(state.points_x[index])
            points_y.append(state.points_y[index])
            shapes_x.append(state.shapes_x[index])
            shapes_y.append(state.shapes_y[index])
            dtypes.append(state.dtypes[index])
            scales.append(state.scales[index])
        new_state = measure.State(state.batch, state.M, state.N, state.K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, 'float32')
        return new_state



class kFuseRule(SearchRules):
    def __init__(self, rule_name = "kfuse_rule"):
        super().__init__(rule_name)
        self.index = []
        self.type = ''
        self.k_dist = float('inf')

    def kdist(self, state, i, j):
        pk = max(state.points_y[i][1], state.points_y[j][1])
        sk = min(state.points_y[i][1] + state.shapes_y[i][2], state.points_y[j][1] + state.shapes_y[j][2])
        if(pk - sk >= 0):
            return pk - sk
        else:
            return -1

    def meetcondition(self, state):
        if(len(state.points_y) < 2):
            return False
        for i in range(0, len(state.points_y)):
            pn1, pk1 = state.points_y[i]
            sn1, sk1 = state.shapes_y[i][-2:]
            for j in range(i + 1, len(state.points_y)):
                pn2, pk2 = state.points_y[j]
                sn2, sk2 = state.shapes_y[j][-2:]
                if(pn1 == pn2):
                    if(sn1 == sn2):
                        dist = self.kdist(state, i, j)
                        if(self.k_dist <= dist):
                            break
                        else: 
                            self.type = 'n1'
                            self.index = [i, j]
                            self.k_dist = dist
                    else:
                        dist = self.kdist(state, i, j)
                        if(self.k_dist <= dist):
                            break
                        else:
                            self.type = 'n2_front'
                            self.index = [i, j]
                            self.k_dist = dist
                else:
                    if(pn1 + sn1 == pn2 + sn2):
                        dist = self.kdist(state, i, j)
                        if(self.k_dist <= dist):
                            break
                        else: 
                            self.type = 'n2_back'
                            self.index = [i, j]
                            self.k_dist = dist
                    else:
                        if((pn1 > pn2 and pn1 < pn2 + sn2 and pn1 + sn1 > pn2 + sn2) or (pn1 < pn2 and pn2 < pn1 + sn1 and pn1 + sn1 < pn2 + sn2)):
                            dist = self.kdist(state, i, j)
                            if(self.k_dist <= dist):
                                break
                            else:
                                self.type = 'n3_out'
                                self.index = [i, j]
                                self.k_dist = dist
                        elif((pn1 < pn2 and pn1 + sn1 > pn2 + sn2) or (pn1 > pn2 and pn1 + sn1 < pn2 + sn2)):
                            dist = self.kdist(state, i, j)
                            if(self.k_dist <= dist):
                                break
                            else:
                                self.type = 'n3_in'
                                self.index = [i, j]
                                self.k_dist = dist
                        else:
                            continue
        if(self.k_dist != float('inf') and self.k_dist != -1):
            return True
        else:
            return False
        

    def apply(self, state):
        new_state = copy.deepcopy(state)
        pn1, pk1 = new_state.points_y[self.index[0]] # xm为0，xk与yk相同
        pn2, pk2 = new_state.points_y[self.index[1]] 
        sn1, sk1 = new_state.shapes_y[self.index[0]][-2:] # sxm为M，sxk与syk相同
        sn2, sk2 = new_state.shapes_y[self.index[1]][-2:]
        b, m, _ = new_state.shapes_x[self.index[0]]
        dtype1 = self.GetDtypeNum(new_state, self.index[0])
        dtype2 = self.GetDtypeNum(new_state, self.index[1])

        if(self.type == 'n1'):
            new_state.points_y[self.index[0]][1] = min(pk1, pk2)
            new_state.points_x[self.index[0]][1] = min(pk1, pk2)
            new_state.shapes_y[self.index[0]][2] = sk1 + sk2 + self.k_dist
            new_state.shapes_x[self.index[0]][2] = sk1 + sk2 + self.k_dist
            new_state.points_y.pop(self.index[1])
            new_state.points_x.pop(self.index[1])
            new_state.shapes_y.pop(self.index[1])
            new_state.shapes_x.pop(self.index[1])
            new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            new_state.dtypes.pop(self.index[1])
            new_state.scales.pop(self.index[1])
            return new_state

        elif(self.type == 'n2_front'):
            new_state.points_y[self.index[0]][1] = min(pk1, pk2)
            new_state.points_x[self.index[0]][1] = min(pk1, pk2)
            new_state.shapes_y[self.index[0]][2] = sk1 + sk2 + self.k_dist
            new_state.shapes_x[self.index[0]][2] = sk1 + sk2 + self.k_dist
            new_state.shapes_y[self.index[0]][1] = min(sn1, sn2)
            new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            if(sn1 < sn2):
                new_state.points_y[self.index[1]][1] = pk2
                new_state.points_x[self.index[1]][1] = pk2
                new_state.points_y[self.index[1]][0] = pn1 + sn1
                new_state.shapes_y[self.index[1]][1] = sn2 - sn1
            else:
                new_state.points_y[self.index[1]][1] = pk1
                new_state.points_x[self.index[1]][1] = pk1
                new_state.points_y[self.index[1]][0] = pn2 + sn2
                new_state.shapes_y[self.index[1]][1] = sn1 - sn2
            return new_state

        elif(self.type == 'n2_back'):
            new_state.points_y[self.index[1]][1] = min(pk1, pk2)
            new_state.points_x[self.index[1]][1] = min(pk1, pk2)
            new_state.shapes_y[self.index[1]][2] = sk1 + sk2 + self.k_dist
            new_state.shapes_x[self.index[1]][2] = sk1 + sk2 + self.k_dist
            new_state.points_y[self.index[1]][0] = max(pn1, pn2)
            new_state.shapes_y[self.index[1]][1] = min(sn1, sn2)
            new_state.dtypes[self.index[1]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[1]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            if(sn1 < sn2):
                new_state.points_y[self.index[0]][1] = pk2
                new_state.points_x[self.index[0]][1] = pk2
                new_state.shapes_y[self.index[0]][2] = sk2
                new_state.shapes_x[self.index[0]][2] = sk2
                new_state.points_y[self.index[0]][0] = pn2
                new_state.shapes_y[self.index[0]][1] = sn2 - sn1
            else:
                new_state.points_y[self.index[0]][1] = pk1
                new_state.points_x[self.index[0]][1] = pk1
                new_state.shapes_y[self.index[0]][2] = sk1
                new_state.shapes_x[self.index[0]][2] = sk1
                new_state.points_y[self.index[0]][0] = pn1
                new_state.shapes_y[self.index[0]][1] = sn1 - sn2
            return new_state

        elif(self.type == 'n3_out'):
            if(pn1 < pn2):
                new_state.shapes_y[self.index[0]][1] = pn2 - pn1
                new_state.points_y[self.index[1]][1] = min(pk1, pk2)
                new_state.points_x[self.index[1]][1] = min(pk1, pk2)
                new_state.shapes_y[self.index[1]][1] = sn1 + pn1 - pn2
                new_state.shapes_y[self.index[1]][2] = sk1 + sk2 + self.k_dist
                new_state.shapes_x[self.index[1]][2] = sk1 + sk2 + self.k_dist
                new_state.dtypes[self.index[1]] = self.GetDtype(max(dtype1, dtype2))
                new_state.scales[self.index[1]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
                point_y_3 = [pn1 + sn1, pk2]
                point_x_3 = [0, pk2]
                shape_y_3 = [b, sn2 - pn1 - sn1 + pn2, sk2]
                shape_x_3 = [b, m, sk2]
                new_state.points_y.insert(self.index[1] + 1, point_y_3)
                new_state.points_x.insert(self.index[1] + 1, point_x_3)
                new_state.shapes_y.insert(self.index[1] + 1, shape_y_3)
                new_state.shapes_x.insert(self.index[1] + 1, shape_x_3)
                new_state.dtypes.insert(self.index[1] + 1, self.GetDtype(dtype2))
                new_state.scales.insert(self.index[1] + 1, self.GetScale(self.GetDtype(dtype2)))
                return new_state
            else:
                new_state.shapes_y[self.index[1]][1] = pn1 - pn2
                new_state.points_y[self.index[0]][1] = min(pk1, pk2)
                new_state.points_x[self.index[0]][1] = min(pk1, pk2)
                new_state.shapes_y[self.index[0]][1] = sn2 + pn2 - pn1
                new_state.shapes_y[self.index[0]][2] = sk1 + sk2 + self.k_dist
                new_state.shapes_x[self.index[0]][2] = sk1 + sk2 + self.k_dist
                new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
                new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
                point_y_3 = [pn2 + sn2, pk1]
                point_x_3 = [0, pk1]
                shape_y_3 = [b, sn1 - pn2 - sn2 + pn1, sk1]
                shape_x_3 = [b, m, sk1]
                new_state.points_y.insert(self.index[0] + 1, point_y_3)
                new_state.points_x.insert(self.index[0] + 1, point_x_3)
                new_state.shapes_y.insert(self.index[0] + 1, shape_y_3)
                new_state.shapes_x.insert(self.index[0] + 1, shape_x_3)
                new_state.dtypes.insert(self.index[0] + 1, self.GetDtype(dtype1))
                new_state.scales.insert(self.index[0] + 1, self.GetScale(self.GetDtype(dtype1)))
                return new_state

        elif(self.type == 'n3_in'):
            if(pn1 < pn2):
                new_state.shapes_y[self.index[0]][1] = pn2 - pn1
                new_state.points_y[self.index[1]][1] = min(pk1, pk2)
                new_state.points_x[self.index[1]][1] = min(pk1, pk2)
                new_state.shapes_y[self.index[1]][2] = sk1 + sk2 + self.k_dist
                new_state.shapes_x[self.index[1]][2] = sk1 + sk2 + self.k_dist
                new_state.dtypes[self.index[1]] = self.GetDtype(max(dtype1, dtype2))
                new_state.scales[self.index[1]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
                point_y_3 = [pn2 + sn2, pk1]
                point_x_3 = [0, pk1]
                shape_y_3 = [b, sn1 + pn1 - pn2 - sn2 , sk1]
                shape_x_3 = [b, m, sk1]
                new_state.points_y.insert(self.index[1] + 1, point_y_3)
                new_state.points_x.insert(self.index[1] + 1, point_x_3)
                new_state.shapes_y.insert(self.index[1] + 1, shape_y_3)
                new_state.shapes_x.insert(self.index[1] + 1, shape_x_3)
                new_state.dtypes.insert(self.index[1] + 1, self.GetDtype(dtype1))
                new_state.scales.insert(self.index[1] + 1, self.GetScale(self.GetDtype(dtype1)))
                return new_state
            else:
                new_state.shapes_y[self.index[1]][1] = pn1 - pn2
                new_state.points_y[self.index[0]][1] = min(pk1, pk2)
                new_state.points_x[self.index[0]][1] = min(pk1, pk2)
                new_state.shapes_y[self.index[0]][2] = sk1 + sk2 + self.k_dist
                new_state.shapes_x[self.index[0]][2] = sk1 + sk2 + self.k_dist
                new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
                new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
                point_y_3 = [pn1 + sn1, pk2]
                point_x_3 = [0, pk2]
                shape_y_3 = [b, sn2 - pn1 - sn1 + pn2, sk2]
                shape_x_3 = [b, m, sk2]
                new_state.points_y.insert(self.index[0] + 1, point_y_3)
                new_state.points_x.insert(self.index[0] + 1, point_x_3)
                new_state.shapes_y.insert(self.index[0] + 1, shape_y_3)
                new_state.shapes_x.insert(self.index[0] + 1, shape_x_3)
                new_state.dtypes.insert(self.index[0] + 1, self.GetDtype(dtype2))
                new_state.scales.insert(self.index[0] + 1, self.GetScale(self.GetDtype(dtype2)))
                return new_state
        
        else:
            return new_state
        




class nFuseRule(SearchRules):
    def __init__(self, rule_name = "nfuse_rule"):
        super().__init__(rule_name)
        self.index = []
        self.type = ''
        self.n_dist = float('inf')

    def ndist(self, state, i, j):
        pn = max(state.points_y[i][0], state.points_y[j][0])
        sn = min(state.points_y[i][0] + state.shapes_y[i][1], state.points_y[j][0] + state.shapes_y[j][1])
        if(pn - sn >= 0):
            return pn - sn
        else:
            return -1        


    def meetcondition(self, state):
        if(len(state.points_y) < 2):
            return False
        for i in range(0, len(state.points_y)):
            pn1, pk1 = state.points_y[i]
            sn1, sk1 = state.shapes_y[i][-2:]
            for j in range(i + 1, len(state.points_y)):
                pn2, pk2 = state.points_y[j]
                sn2, sk2 = state.shapes_y[j][-2:]
                if(pk1 == pk2):
                    if(sk1 == sk2):
                        dist = self.ndist(state, i, j)
                        if(self.n_dist <= dist):
                            break
                        else: 
                            self.type = 'k1'
                            self.index = [i, j]
                            self.n_dist = dist
                    else:
                        dist = self.ndist(state, i, j)
                        if(self.n_dist <= dist):
                            break
                        else:
                            self.type = 'k2_front'
                            self.index = [i, j]
                            self.n_dist = dist
                else:
                    if(pk1 + sk1 == pk2 + sk2):
                        dist = self.ndist(state, i, j)
                        if(self.n_dist <= dist):
                            break
                        else: 
                            self.type = 'k2_back'
                            self.index = [i, j]
                            self.n_dist = dist
                    else:
                        if((pk1 > pk2 and pk1 < pk2 + sk2 and pk1 + sk1 > pk2 + sk2) or (pk1 < pk2 and pk2 < pk1 + sk1 and pk1 + sk1 < pk2 + sk2)):
                            dist = self.ndist(state, i, j)
                            if(self.n_dist <= dist):
                                break
                            else:
                                self.type = 'k3_out'
                                self.index = [i, j]
                                self.n_dist = dist
                        elif((pk1 < pk2 and pk1 + sk1 > pk2 + sk2) or (pk1 > pk2 and pk1 + sk1 < pk2 + sk2)):
                            dist = self.ndist(state, i, j)
                            if(self.n_dist <= dist):
                                break
                            else:
                                self.type = 'k3_in'
                                self.index = [i, j]
                                self.n_dist = dist
                        else:
                            continue
        if(self.n_dist != float('inf') and self.n_dist != -1):
            return True
        else:
            return False          


    def apply(self, state):
        new_state = copy.deepcopy(state)
        pn1, pk1 = new_state.points_y[self.index[0]] # xm为0，xk与yk相同
        pn2, pk2 = new_state.points_y[self.index[1]] 
        sn1, sk1 = new_state.shapes_y[self.index[0]][-2:] # sxm为M，sxk与syk相同
        sn2, sk2 = new_state.shapes_y[self.index[1]][-2:]
        b, m, _ = new_state.shapes_x[self.index[0]]
        dtype1 = self.GetDtypeNum(new_state, self.index[0])
        dtype2 = self.GetDtypeNum(new_state, self.index[1])

        if(self.type == 'k1'):
            new_state.points_y[self.index[0]][0] = min(pn1, pn2)
            new_state.shapes_y[self.index[0]][1] = sn1 + sn2 + self.n_dist
            new_state.points_y.pop(self.index[1])
            new_state.points_x.pop(self.index[1])
            new_state.shapes_y.pop(self.index[1])
            new_state.shapes_x.pop(self.index[1])
            new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            new_state.dtypes.pop(self.index[1])
            new_state.scales.pop(self.index[1])
            return new_state

        elif(self.type == 'k2_front'):
            new_state.points_y[self.index[0]][0] = min(pn1, pn2)
            new_state.shapes_y[self.index[0]][1] = sn1 + sn2 + self.n_dist
            new_state.shapes_y[self.index[0]][2] = max(sk1, sk2)
            new_state.shapes_x[self.index[0]][2] = max(sk1, sk2)
            new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            new_state.points_y.pop(self.index[1])
            new_state.points_x.pop(self.index[1])
            new_state.shapes_y.pop(self.index[1])
            new_state.shapes_x.pop(self.index[1])
            new_state.dtypes.pop(self.index[1])
            new_state.scales.pop(self.index[1])

            return new_state

        elif(self.type == 'k2_back'):
            new_state.points_y[self.index[0]][1] = min(pk1, pk2)
            new_state.points_x[self.index[0]][1] = min(pk1, pk2)
            new_state.shapes_y[self.index[0]][2] = max(sk1, sk2)
            new_state.shapes_x[self.index[0]][2] = max(sk1, sk2)
            new_state.points_y[self.index[0]][0] = min(pn1, pn2)
            new_state.shapes_y[self.index[0]][1] = sn1 + sn2 + self.n_dist
            new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            new_state.points_y.pop(self.index[1])
            new_state.points_x.pop(self.index[1])
            new_state.shapes_y.pop(self.index[1])
            new_state.shapes_x.pop(self.index[1])
            new_state.dtypes.pop(self.index[1])
            new_state.scales.pop(self.index[1])
            return new_state

        elif(self.type == 'k3_out'):
            new_state.points_y[self.index[0]][0] = min(pn1, pn2)
            new_state.points_y[self.index[0]][1] = min(pk1, pk2)
            new_state.points_x[self.index[0]][1] = min(pk1, pk2)
            new_state.shapes_y[self.index[0]][1] = sn1 + sn2 + self.n_dist
            new_state.shapes_y[self.index[0]][2] = max(pk1 + sk1, pk2 + sk2) - min(pk1, pk2)
            new_state.shapes_x[self.index[0]][2] = max(pk1 + sk1, pk2 + sk2) - min(pk1, pk2)
            new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            new_state.points_y.pop(self.index[1])
            new_state.points_x.pop(self.index[1])
            new_state.shapes_y.pop(self.index[1])
            new_state.shapes_x.pop(self.index[1])
            new_state.dtypes.pop(self.index[1])
            new_state.scales.pop(self.index[1])
            return new_state

        elif(self.type == 'k3_in'):
            new_state.points_y[self.index[0]][0] = min(pn1, pn2)
            new_state.points_y[self.index[0]][1] = min(pk1, pk2)
            new_state.points_x[self.index[0]][1] = min(pk1, pk2)
            new_state.shapes_y[self.index[0]][1] = sn1 + sn2 + self.n_dist
            new_state.shapes_y[self.index[0]][2] = max(sk1, sk2)
            new_state.shapes_x[self.index[0]][2] = max(sk1, sk2)
            new_state.dtypes[self.index[0]] = self.GetDtype(max(dtype1, dtype2))
            new_state.scales[self.index[0]] = self.GetScale(self.GetDtype(max(dtype1, dtype2)))
            new_state.points_y.pop(self.index[1])
            new_state.points_x.pop(self.index[1])
            new_state.shapes_y.pop(self.index[1])
            new_state.shapes_x.pop(self.index[1])
            new_state.dtypes.pop(self.index[1])
            new_state.scales.pop(self.index[1])
            return new_state
            
        else:
            return new_state

        



if __name__ == "__main__":
    state = measure.State(16, 512, 512, 512, [[0, 0], [0, 64], [0, 0]], 
                     [[0, 0], [0, 64], [256, 0]], [[4, 16, 64], [4, 16, 64], [4, 16, 64]], [[4, 128, 64], [4, 128, 64], [4, 256, 64]], 
                     ["int4", 'int8', 'int16'], [8.0/2**3, 8.0/2**7, 8.0/2**15], "float32")
    
    new_state = SortRule().apply(state)
    print(new_state.ToStr())
    k = kFuseRule()
    if(k.meetcondition(new_state)):
        print(k.index)
        s = k.apply(new_state)
        print(s.ToStr())