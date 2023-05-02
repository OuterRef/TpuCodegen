import json
import argparse
from typing import List


####################################
####       Data Structure       ####
####################################
class csinnTensor():
    def __init__(self, name: str, shape: List[int], layout: str, is_const: int,
                 quant_channel: int, dtype: str, **kwargs) -> None:
        self.name = name
        self.shape = [str(dim) for dim in shape]
        self.dim_count = str(len(self.shape))
        self.layout = layout
        self.is_const = str(is_const)
        self.quant_channel = str(quant_channel)
        self.dtype = dtype
        # TODO: real data
        self.data = f"malloc({' * '.join(self.shape)} * sizeof(float))"
        self.additional = kwargs

    def getDefinitionLine(self) -> str:
        return f"struct csinn_tensor *{self.name} = csinn_alloc_tensor(sess);\n"

    def getAttributeLines(self) -> List[str]:
        lines = []
        for i in range(len(self.shape)):
            lines.append(f"{self.name}->dim[{i}] = {self.shape[i]}")
        for k, v in self.__dict__.items():
            if k == 'name' or k == 'shape':
                continue
            elif k == 'additional':
                for ad_k, ad_v in v.items():
                    lines.append(f"{self.name}->{ad_k} = {ad_v}")
            else:
                lines.append(f"{self.name}->{k} = {v}")
        lines = [line + ";\n" for line in lines]
        return lines
    

class csinnParam():
    def __init__(self, param_type: str, name: str, **kwargs) -> None:
        self.param_type = param_type
        self.name = name
        self.attr = kwargs
    
    def getDefinitionLine(self):
        return f"struct {self.param_type} *{self.name} = csinn_alloc_params(sizeof(struct {self.param_type}), sess);\n"
    
    def getAttributeLines(self) -> List[str]:
        lines = []
        for k, v in self.attr.items():
            if type(v) is list:
                lines.append(f"int {k}[{len(v)}] = {{{', '.join([str(item) for item in v])}}}")
                lines.append(f"{self.name}->{k} = {k}")
            else:
                lines.append(f"{self.name}->{k} = {v}")
        lines = [line + ";\n" for line in lines]
        return lines


class genModel():
    def __init__(self, json_file):
        self.json_file = json_file
        self.code = []       # code lines for definitions
        self.exec_code = []  # code lines for graph execution
        self.value_list = [] # values that have been defined
        self.opcode_table = None
        self.run()

    def run(self):
        self.genSession()
        self.opcode_table = self.opCodeTable()
        for op in self.json_file:
            self.genOpCode(op)

    def genSession(self):
        ''' Init session '''
        self.code.append("struct csinn_session *sess = csinn_alloc_session();\n")
        self.code.append("sess->base_run_mode = CSINN_RM_LAYER;\n")
    
    def genOpCode(self, op):
        opcode = op["opcode"]
        try:
            self.opcode_table[opcode](op)
        except KeyError as e:
            print(f"[Warning] {e} is not implemented, ignoring...")

    def emitC(self):
        self.exec_code.insert(0, "uint64_t start_time, end_time;\n")
        self.exec_code.insert(1, "start_time = shl_get_timespec();\n")
        self.code += self.exec_code
        self.code.append("end_time = shl_get_timespec();\n")
        self.code.append(r'printf("Run graph execution time: %.5fms, FPS=%.2f\n", ((float)(end_time - start_time)) / 1000000, 1000000000.0 / ((float)(end_time - start_time)));' + '\n')
        self.code.append("return 0;\n")
        self.code = ["\t" + line for line in self.code]
        self.code.insert(0, "#include <shl_ref.h>\n\nint main(int argc, char **argv)\n{\n")
        self.code.append("}\n")
        with open("model.c", "w") as f:
            for line in self.code:
                f.write(line)

    def opCodeTable(self) -> dict:
        ''' opcode : genCodeMethod '''
        return {
            "tpu.Conv2D" : self.conv2D,
            "tpu.Permute": self.transpose,
            "tpu.Pad"    : self.pad,
        }
    
    ####################################
    ####         Operations         ####
    ####################################
    def conv2D(self, op):
        # params
        params = op["attributes"]
        param_name = "line_" + str(op["file-line"]) + "_param"
        conv2d_param = csinnParam("csinn_conv2d_params",
                                  param_name,
                                  stride_height=params["strides"][0],
                                  stride_width=params["strides"][1],
                                  pad_left=params["pads"][0],
                                  pad_right=params["pads"][1],
                                  pad_top=params["pads"][2],
                                  pad_down=params["pads"][3],
                                  dilation_width=params["dilations"][0],
                                  dilation_height=params["dilations"][1],
                                  group=1)
        self.code.append(conv2d_param.getDefinitionLine())
        self.code += conv2d_param.getAttributeLines()
        self.code.append(f'{param_name}->base.layout = CSINN_LAYOUT_NCHW;\n')
        self.code.append(f'{param_name}->conv_extra.fuse_zp2bias = false;\n')
        self.code.append(f'{param_name}->base.api = CSINN_C906;\n')
        
        # values
        operands = op["operands"]
        results = op["results"]
        names = []
        for idx, value in enumerate(operands+results):
            if not value:
                continue
            name = "value_" + value["name"]
            names.append(name)
            if name in self.value_list:
                continue
            self.value_list.append(name)
            shape = getShapeFromType(value["type"])
            if idx == 1:
                shape = [shape[1], shape[2], params["kernel_shape"][0], params["kernel_shape"][1]]
            value_tensor = csinnTensor(name, shape, "CSINN_LAYOUT_NCHW", 0, 1, "CSINN_DTYPE_FLOAT32")
            self.code.append(value_tensor.getDefinitionLine())
            self.code += value_tensor.getAttributeLines()
        name_series = f"{names[0]}, {names[3]}, {names[1]}, {names[2]}, {param_name}"
        self.code.append(f'csinn_conv2d_init({name_series});\n')
        self.exec_code.append(f'csinn_conv2d({name_series});\n')

    def transpose(self, op):
        # params
        params = op["attributes"]
        param_name = "line_" + str(op["file-line"]) + "_param"
        permute_param = csinnParam("csinn_transpose_params",
                                   param_name,
                                   permute_num=len(params["order"]),
                                   permute=params["order"])
        self.code.append(permute_param.getDefinitionLine())
        self.code += permute_param.getAttributeLines()

        # values
        operands = op["operands"]
        results = op["results"]
        names = []
        for value in (operands+results):
            if not value:
                continue
            name = "value_" + value['name']
            names.append(name)
            if name in self.value_list:
                continue
            self.value_list.append(name)
            shape = getShapeFromType(value["type"])
            value_tensor = csinnTensor(name, shape, "CSINN_LAYOUT_NULL", 0, 1, "CSINN_DTYPE_FLOAT32")
            self.code.append(value_tensor.getDefinitionLine())
            self.code += value_tensor.getAttributeLines()
        name_series = f"{names[0]}, {names[1]}, {param_name}"
        self.code.append(f"csinn_transpose_init({name_series});\n")
        self.exec_code.append(f"csinn_transpose({name_series});\n")

    def pad(self, op):
        # params
        params = op["attributes"]
        param_name = "line_" + str(op["file-line"]) + "_param"
        pad_len = len(params["paddings"])
        pad_param = csinnParam("csinn_pad_params",
                               param_name,
                               pad_value=params["val"],
                               pad_mode="CSINN_PAD_CONSTANT",
                               pad_before=params["paddings"][:pad_len // 2],
                               pad_after=params["paddings"][pad_len // 2:])
        self.code.append(pad_param.getDefinitionLine())
        self.code += pad_param.getAttributeLines()

        # values
        operands = op["operands"]
        results = op["results"]
        names = []
        for value in (operands+results):
            if not value:
                continue
            name = "value_" + value['name']
            names.append(name)
            if name in self.value_list:
                continue
            self.value_list.append(name)
            shape = getShapeFromType(value['type'])
            value_tensor = csinnTensor(name, shape, "CSINN_LAYOUT_NULL", 0, 1, "CSINN_DTYPE_FLOAT32")
            self.code.append(value_tensor.getDefinitionLine())
            self.code += value_tensor.getAttributeLines()
        name_series = f"{names[0]}, {names[1]}, {param_name}"
        self.code.append(f"csinn_pad_init({name_series});\n")
        self.exec_code.append(f"csinn_pad({name_series});\n")


#####################################
####          Utilities          ####
#####################################
def getShapeFromType(type: str) -> List[int]:
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
