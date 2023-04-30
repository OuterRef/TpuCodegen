#loc = loc(unknown)
#loc1 = loc("input")
module attributes {module.FLOPs = 751400000 : i64, module.asymmetric = true, module.chip = "bm1684x", module.coeff_addr = 4294967296 : i64, module.coeff_size = 45056 : i64, module.mode = "F32", module.name = "Conv2d_2", module.neuron_addr = 4295012352 : i64, module.neuron_size = 12963840 : i64, module.platform = "ONNX", module.state = "TPU_ADDRESSED", module.weight_file = "conv2d_2_tpu_addressed_bm1684x_f32_weight.npz"} {
  func.func @main(%arg0: tensor<4x16x100x100xf32> loc(unknown)) -> tensor<4x65x100x100xf32, 4297572352 : i64> {
    %0 = "top.Input"(%arg0) : (tensor<4x16x100x100xf32>) -> tensor<4x16x100x100xf32, 4295012352 : i64> loc(#loc1)
    %1 = call @subfunc_0(%0) : (tensor<4x16x100x100xf32, 4295012352 : i64>) -> tensor<4x65x100x100xf32, 4297572352 : i64> loc(#loc)
    return %1 : tensor<4x65x100x100xf32, 4297572352 : i64> loc(#loc)
  } loc(#loc)
  func.func @subfunc_0(%arg0: tensor<4x16x100x100xf32, 4295012352 : i64> loc("input")) -> tensor<4x65x100x100xf32, 4297572352 : i64> attributes {id = 0 : i64, mode = #tpu<run_mode TPU_STATIC>} {
    %0 = "top.Weight"() : () -> tensor<1x65x16x9xf32, 4294967296 : i64> loc(#loc2)
    %1 = "top.Weight"() : () -> tensor<1x65x1x1xf32, 4295008256 : i64> loc(#loc3)
    %2 = "tpu.Conv2D"(%arg0, %0, %1) {coeff_merged = false, dilations = [1, 1], do_relu = false, group = 1 : i64, kernel_shape = [3, 3], kernel_zp = 0 : i64, pads = [1, 1, 1, 1], quant_mode = #tpu<rq_mode MultiplierShift>, relu_limit = -1.000000e+00 : f64, strides = [1, 1], use_3ic_optimize = 0 : i64, with_bias = true} : (tensor<4x16x100x100xf32, 4295012352 : i64>, tensor<1x65x16x9xf32, 4294967296 : i64>, tensor<1x65x1x1xf32, 4295008256 : i64>) -> tensor<4x65x100x100xf32, 4297572352 : i64> loc(#loc4)
    return %2 : tensor<4x65x100x100xf32, 4297572352 : i64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("weight")
#loc3 = loc("bias")
#loc4 = loc("output_Conv")

