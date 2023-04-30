import json
import argparse

class genModel():
    def __init__(self, json_file):
        self.json_file = json_file
        self.code = []       # code lines for definitions
        self.exec_code = []  # code lines for graph execution
        self.opcode_table = None
        self.run()

    def run(self):
        self.genSession()
        self.opcode_table = self.opCodeTable()
        for op in self.json_file:
            self.genOpCode(op)

    def genSession(self):
        ''' Init session '''
        self.addCode("struct csinn_session *sess = csinn_alloc_session();\n")
        self.addCode("sess->base_run_mode = CSINN_RM_LAYER;\n")
    
    def genOpCode(self, op):
        opcode = op["opcode"]
        self.opcode_table[opcode](op)

    def addCode(self, string: str):
        self.code.append(string)

    def emitC(self):
        self.exec_code.insert(0, "uint64_t start_time, end_time;\n")
        self.exec_code.insert(1, "start_time = shl_get_timespec();\n")
        self.code += self.exec_code
        self.addCode("end_time = shl_get_timespec();\n")
        self.addCode(r'printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time - start_time)) / 1000000, 1000000000.0 / ((float)(end_time - start_time)));' + '\n')
        self.addCode("return 0;\n")
        self.code = ["\t" + line for line in self.code]
        self.code.insert(0, "#include <shl_ref.h>\n\nint main(int argc, char **argv)\n{\n")
        self.addCode("}\n")
        with open("model.c", "w") as f:
            for line in self.code:
                f.write(line)

    def opCodeTable(self) -> dict:
        ''' opcode : genCodeMethod '''
        return {
            "tpu.Conv2D" : self.Conv2D,
        }
    
    ####################################
    ####         Operations         ####
    ####################################
    def Conv2D(self, op):
        params = op["attributes"]
        param_name = "line_" + str(op["file-line"]) + "_param"
        operands = op["operands"]
        results = op["results"]
        self.addCode(f"struct csinn_conv2d_params *{param_name} = csinn_alloc_params(sizeof(struct csinn_conv2d_params), sess);\n")
        self.addCode(f'{param_name}->stride_height = {params["strides"][0]};\n')
        self.addCode(f'{param_name}->stride_width = {params["strides"][1]};\n')
        self.addCode(f'{param_name}->pad_left = {params["pads"][0]};\n')
        self.addCode(f'{param_name}->pad_right = {params["pads"][1]};\n')
        self.addCode(f'{param_name}->pad_top = {params["pads"][2]};\n')
        self.addCode(f'{param_name}->pad_down = {params["pads"][3]};\n')
        self.addCode(f'{param_name}->dilation_width = {params["dilations"][0]};\n')
        self.addCode(f'{param_name}->dilation_height = {params["dilations"][1]};\n')
        self.addCode(f'{param_name}->base.layout = CSINN_LAYOUT_NCHW;\n')
        self.addCode(f'{param_name}->group = 1;\n')
        self.addCode(f'{param_name}->conv_extra.fuse_zp2bias = false;\n')
        self.addCode(f'{param_name}->base.api = CSINN_C906;\n')
        
        names = []
        for idx, value in enumerate(operands+results):
            name = value["name"]
            names.append(name)
            shape = getShapeFromType(value["type"])
            n_dim = len(shape)
            self.addCode(f"struct csinn_tensor *{name} = csinn_alloc_tensor(sess);\n")
            if idx == 1:
                # weight shape adjustment
                kernel_size = params["kernel_shape"]
                self.addCode(f"{name}->dim[0] = {shape[1]};\n")
                self.addCode(f"{name}->dim[1] = {shape[2]};\n")
                self.addCode(f"{name}->dim[2] = {kernel_size[0]};\n")
                self.addCode(f"{name}->dim[3] = {kernel_size[1]};\n")
            else:
                for i in range(n_dim):
                    self.addCode(f"{name}->dim[{i}] = {shape[i]};\n")
            self.addCode(f"{name}->dim_count = {n_dim};\n")
            self.addCode(f"{name}->layout = CSINN_LAYOUT_NCHW;\n")
            self.addCode(f"{name}->is_const = 0;\n")
            self.addCode(f"{name}->quant_channel = 1;\n")
            self.addCode(f"{name}->dtype = CSINN_DTYPE_FLOAT32;\n")
            # TODO: random initialize, replace with real data in the future
            self.addCode(f"{name}->data = malloc({' * '.join([str(dim) for dim in shape])} * sizeof(float));\n")

        name_series = f"{names[0]}, {names[3]}, {names[1]}, {names[2]}, {param_name}"
        self.addCode(f'csinn_conv2d_init({name_series});\n')
        self.exec_code.append(f'csinn_conv2d({name_series});\n')


def getShapeFromType(type: str):
    ''' Get shape array from tensortype '''
    assert type[:7] == "tensor<"
    shape_str = type.split("<", 1)[-1].split(",")[0]
    shape = shape_str.split("x")[:-1]
    shape = [int(dim) for dim in shape]
    return shape

def mlirAttr2kwargs(attr_str: str):
    ''' Turn mlir-style attr into kwargs. '''
    def getKVpairs(attr_str: str):
        ''' Split by comma, except for lists. '''
        ret = []
        temp = ""
        in_list = 0
        for c in attr_str:
            if c == "[":
                in_list += 1
            if c == "]":
                in_list -= 1
            if c == "," and in_list == 0:
                ret.append(temp)
                temp = ""
                continue
            temp += c
        ret.append(temp)
        return ret
    kwargs = {}
    for kv in getKVpairs(attr_str):
        kv = kv.strip().split(" : ")[0].split(" = ")
        k, v = kv
        kwargs[k] = evalString(v)
    return kwargs

def evalString(string: str):
    ''' 
    Evaluate a string as the type it should be.
    e.g.
        '"ONNX"' -> 'ONNX'
        'true' -> True
        '0' -> 0
    '''
    try:
        # -> numeric
        return eval(string)
    except:
        # -> string
        if string[0] == '\"' and string[-1] == '\"':
            return string[1:-2]
        # -> bool
        elif string == "true":
            return True
        elif string == "false":
            return False
        else:
            print(f"[Warning] Value {string} not expected, return as string.")
            return string

def cmdArgs():
    parser = argparse.ArgumentParser(description='MLIR file parser')
    parser.add_argument(
        "-m",
        "--mlir_file", 
        type=str, 
        required=True,
        help="MLIR file to be parsed"
    )
    parser.add_argument(
        "-j",
        "--json_file", 
        type=str, 
        required=True,
        help="bmodel JSON file"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cmdArgs()
    with open(args.json_file, "r") as jf:
        json_file = json.load(jf)
    with open(args.mlir_file, "r") as f:
        mlir_file = f.readlines()
    # add op attributes into json file
    for op in json_file:
        line_idx = op["file-line"]
        mlir_line = mlir_file[line_idx - 1].strip()
        mlir_attr = mlir_line.split("{", 1)[-1].split("}", 1)[0]
        op["attributes"] = mlirAttr2kwargs(mlir_attr)
    
    # generate model.c
    print("-"*20)
    model = genModel(json_file)
    model.emitC()

    

