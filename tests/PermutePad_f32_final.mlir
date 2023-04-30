#loc = loc(unknown)
#loc1 = loc("in_0")
module attributes {module.FLOPs = 0 : i64, module.asymmetric = true, module.chip = "bm1684x", module.coeff_addr = 4294967296 : i64, module.coeff_size = 0 : i64, module.mode = "F32", module.name = "PermutePad", module.neuron_addr = 4294967296 : i64, module.neuron_size = 3117056 : i64, module.platform = "ONNX", module.state = "TPU_ADDRESSED", module.weight_file = "permutepad_tpu_addressed_bm1684x_f32_weight.npz"} {
  func.func @main(%arg0: tensor<1x10x20x30x40xf32> loc(unknown)) -> tensor<1x10x32x42x20xf32, 4297007104 : i64> {
    %0 = "top.Input"(%arg0) : (tensor<1x10x20x30x40xf32>) -> tensor<1x10x20x30x40xf32, 4294967296 : i64> loc(#loc1)
    %1 = call @subfunc_0(%0) : (tensor<1x10x20x30x40xf32, 4294967296 : i64>) -> tensor<1x10x32x42x20xf32, 4297007104 : i64> loc(#loc)
    return %1 : tensor<1x10x32x42x20xf32, 4297007104 : i64> loc(#loc)
  } loc(#loc)
  func.func @subfunc_0(%arg0: tensor<1x10x20x30x40xf32, 4294967296 : i64> loc("in_0")) -> tensor<1x10x32x42x20xf32, 4297007104 : i64> attributes {id = 0 : i64, mode = #tpu<run_mode TPU_STATIC>} {
    %0 = "top.None"() : () -> none loc(#loc)
    %1 = "tpu.Pad"(%arg0, %0, %0) {mode = 0 : i64, paddings = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1], val = 0.000000e+00 : f64} : (tensor<1x10x20x30x40xf32, 4294967296 : i64>, none, none) -> tensor<1x10x20x32x42xf32, 4295929856 : i64> loc(#loc2)
    %2 = "tpu.Permute"(%1, %0) {order = [0, 1, 3, 4, 2]} : (tensor<1x10x20x32x42xf32, 4295929856 : i64>, none) -> tensor<1x10x32x42x20xf32, 4297007104 : i64> loc(#loc3)
    return %2 : tensor<1x10x32x42x20xf32, 4297007104 : i64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("in_0_pad")
#loc3 = loc("25_Pad")

