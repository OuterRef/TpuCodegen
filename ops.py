def Conv2D(op):
    code = []
    params = op["attributes"]
    param_name = "line_" + str(op["file-line"]) + "_param"
    operands = op["operands"]
    results = op["results"]
    code.append(f"struct csinn_conv2d_params *{param_name} = csinn_alloc_params(sizeof(struct csinn_conv2d_params), sess);\n")
    code.append(f'{param_name}->stride_height = {params["strides"][0]};\n')
    code.append(f'{param_name}->stride_width = {params["strides"][1]};\n')
    code.append(f'{param_name}->pad_left = {params["pads"][0]};\n')
    code.append(f'{param_name}->pad_right = {params["pads"][1]};\n')
    code.append(f'{param_name}->pad_top = {params["pads"][2]};\n')
    code.append(f'{param_name}->pad_down = {params["pads"][3]};\n')
    code.append(f'{param_name}->dilation_width = {params["dilations"][0]};\n')
    code.append(f'{param_name}->dilation_height = {params["dilations"][1]};\n')
    code.append(f'{param_name}->base.layout = CSINN_LAYOUT_NCHW;\n')
    code.append(f'{param_name}->group = 1;\n')
    code.append(f'{param_name}->conv_extra.fuse_zp2bias = false;\n')
    code.append(f'{param_name}->base.api = CSINN_C906;\n')
    
    names = []
    for idx, value in enumerate(operands+results):
        name = value["name"]
        names.append(name)
        shape = getShapeFromType(value["type"])
        n_dim = len(shape)
        code.append(f"struct csinn_tensor *{name} = csinn_alloc_tensor(sess);\n")
        if idx == 1:
            # weight shape adjustment
            kernel_size = params["kernel_shape"]
            code.append(f"{name}->dim[0] = {shape[1]};\n")
            code.append(f"{name}->dim[1] = {shape[2]};\n")
            code.append(f"{name}->dim[2] = {kernel_size[0]};\n")
            code.append(f"{name}->dim[3] = {kernel_size[1]};\n")
        else:
            for i in range(n_dim):
                code.append(f"{name}->dim[{i}] = {shape[i]};\n")
        code.append(f"{name}->dim_count = {n_dim};\n")
        code.append(f"{name}->layout = CSINN_LAYOUT_NCHW;\n")
        code.append(f"{name}->is_const = 0;\n")
        code.append(f"{name}->quant_channel = 1;\n")
        code.append(f"{name}->dtype = CSINN_DTYPE_FLOAT32;\n")
        # TODO: random initialize, replace with real data in the future
        code.append(f"{name}->data = malloc({' * '.join([str(dim) for dim in shape])} * sizeof(float));\n")

    name_series = f"{names[0]}, {names[3]}, {names[1]}, {names[2]}, {param_name}"
    code.append(f'csinn_conv2d_init({name_series});\n')
    exec_code = [f'csinn_conv2d({name_series});\n']
    return code, exec_code


def getShapeFromType(type: str):
    ''' Get shape array from tensortype '''
    assert type[:7] == "tensor<"
    shape_str = type.split("<", 1)[-1].split(",")[0]
    shape = shape_str.split("x")[:-1]
    shape = [int(dim) for dim in shape]
    return shape