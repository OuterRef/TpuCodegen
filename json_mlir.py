import json
import argparse
from ops import *

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
        code, exec_code = self.opcode_table[opcode](op)
        self.code += code
        self.exec_code += exec_code

    def addCode(self, string: str):
        self.code.append(string)

    def emitC(self):
        self.code += self.exec_code
        self.addCode(r'printf("Run graph completed.\n");' + '\n')
        self.addCode("return 0;\n")
        self.code = ["\t" + line for line in self.code]
        self.code.insert(0, "#include <shl_ref.h>\n\nint main(int argc, char **argv)\n{\n")
        self.addCode("}\n")
        with open("model.c", "w") as f:
            for line in self.code:
                f.write(line)

    @staticmethod
    def opCodeTable() -> dict:
        ''' opcode : genCodeMethod '''
        return {
            "tpu.Conv2D" : Conv2D,
        }

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

    

