import tvm
from tvm import relay
from tvm.topi.utils import get_const_tuple
import numpy as np
from tvm.contrib import graph_executor
import test_bert_base
from tvm._ffi.runtime_ctypes import DataType

class _TypeFinder(relay.expr_functor.ExprMutator):
    def __init__(self, types):
        super().__init__()
        self.counter = 0
        self.vars = {}
        self.types = types
        self.leave = set()  # some variables are not inputs

    def visit_let(self, let):
        self.leave.add(let.var)
        return super().visit_let(let)

    def visit_function(self, fn):
        self.leave.update(fn.params)
        return super().visit_function(fn)

    def visit(self, expr):
        if expr in self.leave:
            return super().visit(expr)
        if expr in self.vars:
            return self.vars[expr]
        if isinstance(expr, tvm.relay.Var):
            self.vars[expr] = expr
            return expr
        if expr in self.types:
            ty = self.types[expr]
            v = tvm.relay.var(f"_{self.counter}", type_annotation=ty)
            self.counter += 1
            self.vars[expr] = v
            return v
        v = super().visit(expr)
        return v

class infer:
    def __init__(self) -> None:
        self.types = {}

    def infer_type(self, node):
        """A method to infer the type of a relay expression."""
        mod = tvm.IRModule.from_expr(node)
        mod = relay.transform.InferType()(mod)
        entry = mod["main"]
        return entry if isinstance(node, relay.Function) else entry.body
    
    def infer_type1(self, node, mod=None):
        """An incremental method to infer the type of a node in the relay graph."""

        if node in self.types:
            return self.types[node]
        if isinstance(node, relay.Var):
            return node.type_annotation

        tf = _TypeFinder(types=self.types)
        new_node = tf.visit(node)
        fn = relay.Function(list(tf.vars.values()), new_node)
        new_mod = tvm.ir.IRModule({"main": fn})
        if mod is not None:
            new_mod.update(mod)
        new_mod = relay.transform.RemoveUnusedFunctions()(new_mod)
        new_mod = relay.transform.InferType()(new_mod)
        entry = new_mod["main"]
        ty = entry.body.checked_type
        self.types[node] = ty
        return self.types[node]

    def infer_shape(self, inputs, mod=None):
        """A method to get the output type of an intermediate node in the graph."""
        typ = self.infer_type1(inputs)
        if hasattr(typ, "shape"):
            # Regular operator that outputs tensors
            return get_const_tuple(typ.shape)
        # The return type is not a tensor, for example List
        return typ
    
    def record_output_type(self, output):
        if isinstance(output, tuple):
            cleaned_output = [o for o in output if o is not None]
            types = self.infer_type(relay.Tuple(cleaned_output))
            for o, t in zip(cleaned_output, types.fields):
                self.types[o] = t
        elif isinstance(output, relay.Expr):
            self.infer_type(output)

def split_heads(x, num_heads, d_model, Infer):
    # 分割最后一个维度到 (num_heads, depth)
    shape = Infer.infer_shape(x)
    head_dim = d_model // num_heads
    # 改变形状
    x = relay.reshape(x, newshape=[shape[1], shape[0] * num_heads, head_dim])
    # 转置结果使得形状为 (batch_size*head, seq_len, head_dim)
    return relay.transpose(x, axes=[1, 0, 2])


def scaled_dot_product_attention(Q, K, V, d_model, mask):
    d_k = relay.const(float(d_model), dtype="float32")
    Q = relay.divide(Q, relay.sqrt(d_k))
    # K = relay.transpose(K, axes=[0, 2, 1])
    matmul_qk = relay.nn.batch_matmul(Q, K, transpose_b=True)
    
    if mask is not None:
        # 假设mask中填充了非常小的负数（如-1e9）在softmax前应用mask
        matmul_qk += mask
    attn = relay.nn.softmax(matmul_qk)
    attn = relay.squeeze(attn) 
    V = relay.transpose(V, axes=[0, 2, 1])
    output = relay.nn.batch_matmul(attn, V)
    return output, attn

def multi_head_attention(query, key, value, num_heads, d_model, params_map, search_map, Infer, num, cur_layer, mask=None):
    # ... 实现多头注意力机制 ...
    head_dim = d_model // num_heads
    shape = Infer.infer_shape(query)
    for i in range(0, len(search_map["shapes_y"])):
        y = relay.Var("Wq"+str(i)+"_"+str(num)+"layer", relay.TensorType(shape=search_map["shapes_y"][i], dtype=search_map["dtypes"][i]))
        params_map["Wq"+str(i)+"_"+str(num)+"layer"] = y
    
    # 线性层并分割成多头
    if(search_map["mix_type"] == "mix" and search_map["coder"][1] == shape[1] and num == cur_layer):
        y_inputs = []
        for i in range(0, len(search_map["shapes_y"])):
            y_inputs.append(params_map["Wq"+str(i)+"_"+str(num)+"layer"])
        x_inputs = relay.op.tensor.tensorsplit(query, search_map["points_x"], search_map["shapes_x"], search_map["dtypes"], search_map["scales"], "float32")
        x_inputs = x_inputs.astuple()
        Infer.record_output_type(x_inputs)
        Q = relay.op.nn.mix_batch_matmul(x_inputs, y_inputs, "int32", search_map["points_y"], [search_map["batch"], search_map["M"], search_map["N"]])
        Q = relay.cast(Q, "float32")
        scale = relay.const(float(1/(2**31 - 1)), dtype="float32")
        Q = relay.multiply(Q, scale)
        Infer.record_output_type(Q)
    elif(search_map["mix_type"] == "mix_t" and search_map["coder"][1] == shape[1] and num == cur_layer):
        y_inputs = []
        for i in range(0, len(search_map["shapes_y"])):
            y_inputs.append(params_map["Wq"+ str(i)+"_"+str(num)+"layer"])
        x_inputs = relay.op.tensor.tensorsplit(query, search_map["points_x"], search_map["shapes_x"], search_map["dtypes"], search_map["scales"], "float32")
        x_inputs = x_inputs.astuple()
        Infer.record_output_type(x_inputs)
        Q = relay.op.nn.mix_batch_matmul_t(x_inputs, y_inputs, "int32", search_map["points_y"], [search_map["batch"], search_map["M"], search_map["N"]])
        Q = relay.cast(Q, "float32")
        scale = relay.const(float(1/(2**31 - 1)), dtype="float32")
        Q = relay.multiply(Q, scale)
        Infer.record_output_type(Q)
    elif(search_map["mix_type"] == "mix_t48" and search_map["coder"][1] == shape[1] and num == cur_layer):
        y_inputs = []
        for i in range(0, len(search_map["shapes_y"])):
            y_inputs.append(params_map["Wq"+ str(i)+"_"+str(num)+"layer"])
        x_inputs = relay.op.tensor.tensorsplit(query, search_map["points_x"], search_map["shapes_x"], search_map["dtypes"], search_map["scales"], "float32")
        x_inputs = x_inputs.astuple()
        Infer.record_output_type(x_inputs)
        Q = relay.op.nn.mix_batch_matmul_t48(x_inputs, y_inputs, "int32", search_map["points_y"], [search_map["batch"], search_map["M"], search_map["N"]])
        Q = relay.cast(Q, "float32")
        scale = relay.const(float(1/(2**31 - 1)), dtype="float32")
        Q = relay.multiply(Q, scale)
        Infer.record_output_type(Q)
    else:
        Wq = params_map["Wq"+str(num)+"layer"]
        Q = relay.nn.batch_matmul(query, Wq)
    Wk = params_map["Wk"+str(num)+"layer"]
    Wv = params_map["Wv"+str(num)+"layer"]

    K = relay.nn.batch_matmul(key, Wk)
    V = relay.nn.batch_matmul(value, Wv)

    # 分割后形状为(batch_size*head, seq_len, head_dim)
    Q = split_heads(Q, num_heads, d_model, Infer)
    K = split_heads(K, num_heads, d_model, Infer)
    V = split_heads(V, num_heads, d_model, Infer)
  
    # 缩放点积注意力
    attn_output, attn_output_weights = scaled_dot_product_attention(Q, K, V, d_model, mask)
    # print(Infer.infer_shape(attn_output))
    
    # 最后一个线性层
    Wo = params_map["Wo"+str(num)+"layer"]
    Wo = relay.broadcast_to(Wo, [search_map["batch"]*num_heads, head_dim, head_dim])
    Wo = relay.reshape(Wo, newshape=[search_map["batch"]*num_heads, head_dim, head_dim])

    output = relay.nn.batch_matmul(attn_output, Wo)
    return output

def positionwise_feed_forward(data, d_model, dim_feedforward, params_map, search_map, num):
    # 定义权重和偏置
    W1 = params_map["W1"+str(num)+"layer"]
    b1 = params_map["b1"+str(num)+"layer"]
    W2 = params_map["W2"+str(num)+"layer"]
    b2 = params_map["b2"+str(num)+"layer"]


    # 第一个线性变换
    W1 = relay.broadcast_to(W1, [search_map["batch"], dim_feedforward, d_model])
    W1 = relay.reshape(W1, newshape=[search_map["batch"], dim_feedforward, d_model])
    ff1 = relay.nn.batch_matmul(data, W1)
    ff1_bias = relay.nn.bias_add(ff1, b1, axis=-1)

    # ReLU激活函数
    relu = relay.nn.relu(ff1_bias)

    # 第二个线性变换
    W2 = relay.broadcast_to(W2, [search_map["batch"], d_model, dim_feedforward])
    W2 = relay.reshape(W2, newshape=[search_map["batch"], d_model, dim_feedforward])
    ff2 = relay.nn.batch_matmul(relu, W2)
    ff2_bias = relay.nn.bias_add(ff2, b2, axis=-1)

    return ff2_bias


def encoder_layer(enc_input, num_heads, d_model, dim_feedforward, params_map, search_map, Infer, num, cur_layer):
    # 多头自注意力部分
    attn_out = multi_head_attention(enc_input, enc_input, enc_input, num_heads, d_model, params_map, search_map, Infer, num, cur_layer)
    # attn_out = relay.nn.dropout(attn_out, rate=0.1)
    attn_out = relay.reshape(attn_out, Infer.infer_shape(enc_input))
    attn_out = relay.add(attn_out, enc_input)  # 残差连接
    attn_norm = relay.nn.layer_norm(attn_out, params_map["gamma"+str(num)+"layer"], params_map["beta"+str(num)+"layer"])  # 层归一化

    # 前馈神经网络部分
    ffn_out = positionwise_feed_forward(attn_norm, d_model, dim_feedforward, params_map, search_map, num)
    # ffn_out = relay.nn.dropout(ffn_out, rate=0.1)
    ffn_out = relay.add(ffn_out, attn_norm)  # 残差连接
    ffn_norm = relay.nn.layer_norm(ffn_out, params_map["gamma_ffn"+str(num)+"layer"], params_map["beta_ffn"+str(num)+"layer"])  # 层归一化
    
    ffn_norm = relay.broadcast_to(ffn_norm, Infer.infer_shape(enc_input))
    ffn_norm = relay.reshape(ffn_norm, Infer.infer_shape(enc_input))
    return ffn_norm

def bert_base(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map, cur_layer):

    params_map = {}
    head_dim = d_model // num_heads
    params_map["encoder_input"] = relay.Var("encoder_input", relay.TensorType(shape=encoder_input_shape))
    for j in range(num_encoder_layers):
        params_map["W1"+str(j)+"layer"] = relay.Var("W1"+str(j)+"layer", relay.TensorType(shape=[dim_feedforward, d_model]))
        params_map["b1"+str(j)+"layer"] = relay.Var("b1"+str(j)+"layer", relay.TensorType(shape=[dim_feedforward]))
        params_map["W2"+str(j)+"layer"] = relay.Var("W2"+str(j)+"layer", relay.TensorType(shape=[d_model, dim_feedforward]))
        params_map["b2"+str(j)+"layer"] = relay.Var("b2"+str(j)+"layer", relay.TensorType(shape=[d_model]))

        for i in range(0, len(search_map["shapes_y"])):
            y = relay.Var("Wq" + str(i)+str(j)+"layer", relay.TensorType(shape=search_map["shapes_y"][i], dtype=search_map["dtypes"][i]))
            params_map["Wq" + str(i)+str(j)+"layer"] = y
        params_map["Wq"+str(j)+"layer"] = relay.Var("Wq"+str(j)+"layer", relay.TensorType(shape=[encoder_input_shape[0], d_model, d_model]))
        params_map["Wk"+str(j)+"layer"] = relay.Var("Wk"+str(j)+"layer", relay.TensorType(shape=[encoder_input_shape[0], d_model, d_model]))
        params_map["Wv"+str(j)+"layer"] = relay.Var("Wv"+str(j)+"layer", relay.TensorType(shape=[encoder_input_shape[0], d_model, d_model]))
        params_map["Wo"+str(j)+"layer"] = relay.Var("Wo"+str(j)+"layer", relay.TensorType(shape=[head_dim, head_dim]))

        params_map["gamma"+str(j)+"layer"] = relay.Var("gamma"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["beta"+str(j)+"layer"] = relay.Var("beta"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["gamma_ffn"+str(j)+"layer"] = relay.Var("gamma_ffn"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["beta_ffn"+str(j)+"layer"] = relay.Var("beta_ffn"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["gamma1"+str(j)+"layer"] = relay.Var("gamma1"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["beta1"+str(j)+"layer"] = relay.Var("beta1"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["gamma2"+str(j)+"layer"] = relay.Var("gamma2"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["beta2"+str(j)+"layer"] = relay.Var("beta2"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["gamma3"+str(j)+"layer"] = relay.Var("gamma3"+str(j)+"layer", relay.TensorType(shape=[d_model]))
        params_map["beta3"+str(j)+"layer"] = relay.Var("beta3"+str(j)+"layer", relay.TensorType(shape=[d_model]))

    Infer = infer()
    # 编码器
    enc_output = params_map["encoder_input"]
    for num in range(num_encoder_layers):
        enc_output = encoder_layer(enc_output, num_heads, d_model, dim_feedforward, params_map, search_map, Infer, num, cur_layer)
    return enc_output 

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
#准备权重，偏置参数
def prepare_params(encoder_input_shape, num_heads, d_model, dim_feedforward, num_encoder_layers, search_map):
    params = {}
    head_dim = d_model // num_heads
    for i in range(0, len(search_map["shapes_y"])):
        y = tvm.nd.array(get_ref_data(search_map["dtypes"][i], search_map["shapes_y"][i]))
        if(search_map["dtypes"][i] == 'int4'):
            y.handle.contents.dtype = DataType('int4')
        params["Wq" + str(i)] = y
    params["Wk"] = tvm.nd.array(np.random.uniform(size=(encoder_input_shape[0], d_model, d_model)).astype("float32"))
    params["Wv"] = tvm.nd.array(np.random.uniform(size=(encoder_input_shape[0], d_model, d_model)).astype("float32"))

    params["Wo"] = tvm.nd.array(np.random.uniform(size=(head_dim, head_dim)).astype("float32"))

    params["W1"] = tvm.nd.array(np.random.uniform(size=(dim_feedforward, d_model)).astype("float32"))
    params["b1"] = tvm.nd.array(np.random.uniform(size=(dim_feedforward)).astype("float32"))
    params["W2"] = tvm.nd.array(np.random.uniform(size=(d_model, dim_feedforward)).astype("float32"))
    params["b2"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))

    params["final_weight"] = tvm.nd.array(np.random.uniform(size=(1000, d_model)).astype("float32"))

    params["gamma"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))
    params["beta"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))

    params["gamma_ffn"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))
    params["beta_ffn"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))

    params["gamma1"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))
    params["beta1"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))

    params["gamma2"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))
    params["beta2"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))

    params["gamma3"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))
    params["beta3"] = tvm.nd.array(np.random.uniform(size=(d_model)).astype("float32"))

    return params
