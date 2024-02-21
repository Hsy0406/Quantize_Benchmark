import tvm
from tvm import relay
from tvm.topi.utils import get_const_tuple
import numpy as np
from tvm.contrib import graph_executor
import test_bert_base
from timeit import default_timer as timer
import transformer_search


def get_ref_data(dtype, shape):
    if dtype == "int4":
        y = np.random.randint(low=-8, high=7, size=shape)
    elif dtype == "int8":
        y = np.random.randint(low=-128, high=127, size=shape).astype(dtype)
    elif dtype == "int16":
        y = np.random.randint(low=-2**15, high=2**15, size=shape).astype(dtype)
    else:
        y = np.random.randint(low=-2**31, high=2**31, size=shape).astype(dtype)
    return y


def measure(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, cur_layer):
    ftimer_time, timeuse = mix_batch_matmul(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, "float32", cur_layer)

    return ftimer_time, timeuse #越小越好
    
def measure_t(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales):
    ftimer_time, timeuse = mix_batch_matmul_t(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, "float32", cur_layer)

    return ftimer_time, timeuse #越小越好

def measure_t48(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales):
    ftimer_time, timeuse = mix_batch_matmul_t48(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, "float32", cur_layer)
    return ftimer_time, timeuse #越小越好


def mix_batch_matmul(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, input_data, cur_layer):
    num_heads = 12
    d_model = N
    head_dim = d_model // num_heads
    dim_feedforward = d_model * 4
    num_encoder_layers = 12
    encoder_input_shape = (batch, M, d_model)

    search_map = {}
    search_map["batch"] = batch
    search_map["M"] = M
    search_map["N"] = N
    search_map["K"] = K
    search_map["points_x"] = points_x
    search_map["points_y"] = points_y
    search_map["shapes_x"] = shapes_x
    search_map["shapes_y"] = shapes_y
    search_map["dtypes"] = dtypes
    search_map["scales"] = scales
    search_map["mix_type"] = "mix"
    search_map["coder"] = encoder_input_shape
    params = transformer_search.prepare_params(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map)


    # 创建bert-base模型
    output = transformer_search.bert_base(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map, cur_layer)

    
    target = "cuda" 
    dev = tvm.device(target, 0)


    func = relay.Function(relay.analysis.free_vars(output), output)
    func = relay.build_module.bind_params_by_name(func, params)
    mod = tvm.IRModule()
    mod["main"] = func


    with tvm.transform.PassContext(opt_level=3):  # Currently only support opt_level=0
        lib = relay.build(mod, target, params=params)
        m = graph_executor.GraphModule(lib["default"](dev))
        start = timer()
        m.set_input("encoder_input", tvm.nd.array(np.random.uniform(size=encoder_input_shape).astype("float32")))
        for i in range(3):
        # set inputs
            m.run()
        end = timer()
        timeuse = (end - start)/3
        print('mean time' , timeuse)
        ftimer = m.module.time_evaluator("run", dev, number=3, repeat=1)
        print(ftimer().mean)

    return ftimer().mean, timeuse




def mix_batch_matmul_t(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, input_data, cur_layer):

    num_heads = 12
    d_model = N
    head_dim = d_model // num_heads
    dim_feedforward = 3072
    num_encoder_layers = 12
    encoder_input_shape = (batch, M, d_model)

    search_map = {}
    search_map["batch"] = batch
    search_map["M"] = M
    search_map["N"] = N
    search_map["K"] = K
    search_map["points_x"] = points_x
    search_map["points_y"] = points_y
    search_map["shapes_x"] = shapes_x
    search_map["shapes_y"] = shapes_y
    search_map["dtypes"] = dtypes
    search_map["scales"] = scales
    search_map["mix_type"] = "mix_t"
    search_map["coder"] = encoder_input_shape
    params = transformer_search.prepare_params(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map)

    # 创建bert-base模型
    output = transformer_search.bert_base(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map, cur_layer)
    
    target = "cuda" 
    dev = tvm.device(target, 0)

    mod = relay.transform.InferType()
    func = relay.Function(relay.analysis.free_vars(output), output)
    func = relay.build_module.bind_params_by_name(func, params)

    mod = tvm.IRModule()
    mod["main"] = func

    with tvm.transform.PassContext(opt_level=0):  # Currently only support opt_level=0
        lib = relay.build(mod, target, params=params)
        m = graph_executor.GraphModule(lib["default"](dev))
        start = timer()
        m.set_input("encoder_input", tvm.nd.array(np.random.uniform(size=encoder_input_shape).astype("float32")))
        for i in range(3):
        # set inputs
            m.run()
        end = timer()
        timeuse = (end - start)/3
        print('mean time' , timeuse)
        ftimer = m.module.time_evaluator("run", dev, number=3, repeat=1)
        print(ftimer().mean)

    return ftimer().mean, timeuse


def mix_batch_matmul_t48(batch, M, N, K, points_x, points_y, shapes_x, shapes_y, dtypes, scales, input_data, cur_layer):

    num_heads = 12
    d_model = N
    head_dim = d_model // num_heads
    dim_feedforward = 3072
    num_encoder_layers = 12
    encoder_input_shape = (batch, M, d_model)

    search_map = {}
    search_map["batch"] = batch
    search_map["M"] = M
    search_map["N"] = N
    search_map["K"] = K
    search_map["points_x"] = points_x
    search_map["points_y"] = points_y
    search_map["shapes_x"] = shapes_x
    search_map["shapes_y"] = shapes_y
    search_map["dtypes"] = dtypes
    search_map["scales"] = scales
    search_map["mix_type"] = "mix_t48"
    search_map["coder"] = encoder_input_shape
    params = transformer_search.prepare_params(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map)


    # 创建bert-base模型
    output = transformer_search.bert_base(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map, cur_layer)
    
    
    target = "cuda" 
    dev = tvm.device(target, 0)


    mod = relay.transform.InferType()
    func = relay.Function(relay.analysis.free_vars(output), output)
    func = relay.build_module.bind_params_by_name(func, params)

    mod = tvm.IRModule()
    mod["main"] = func

    with tvm.transform.PassContext(opt_level=0):  # Currently only support opt_level=0
        lib = relay.build(mod, target, params=params)
        m = graph_executor.GraphModule(lib["default"](dev))
        start = timer()
        m.set_input("encoder_input", tvm.nd.array(np.random.uniform(size=encoder_input_shape).astype("float32")))
        for i in range(3):
        # set inputs
            m.run()
        end = timer()
        timeuse = (end - start)/3
        print('mean time' , timeuse)
        ftimer = m.module.time_evaluator("run", dev, number=3, repeat=1)
        print(ftimer().mean)

    return ftimer().mean, timeuse
