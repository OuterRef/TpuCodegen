#include <shl_ref.h>

int main(int argc, char **argv)
{
	struct csinn_session *sess = csinn_alloc_session();
	sess->base_run_mode = CSINN_RM_LAYER;
	struct csinn_conv2d_params *line_12_param = csinn_alloc_params(sizeof(struct csinn_conv2d_params), sess);
	line_12_param->stride_height = 1;
	line_12_param->stride_width = 1;
	line_12_param->pad_left = 1;
	line_12_param->pad_right = 1;
	line_12_param->pad_top = 1;
	line_12_param->pad_down = 1;
	line_12_param->dilation_width = 1;
	line_12_param->dilation_height = 1;
	line_12_param->base.layout = CSINN_LAYOUT_NCHW;
	line_12_param->group = 1;
	line_12_param->conv_extra.fuse_zp2bias = false;
	line_12_param->base.api = CSINN_C906;
	struct csinn_tensor *input = csinn_alloc_tensor(sess);
	input->dim[0] = 4;
	input->dim[1] = 16;
	input->dim[2] = 100;
	input->dim[3] = 100;
	input->dim_count = 4;
	input->layout = CSINN_LAYOUT_NCHW;
	input->is_const = 0;
	input->quant_channel = 1;
	input->dtype = CSINN_DTYPE_FLOAT32;
	input->data = malloc(4 * 16 * 100 * 100 * sizeof(float));
	struct csinn_tensor *weight = csinn_alloc_tensor(sess);
	weight->dim[0] = 65;
	weight->dim[1] = 16;
	weight->dim[2] = 3;
	weight->dim[3] = 3;
	weight->dim_count = 4;
	weight->layout = CSINN_LAYOUT_NCHW;
	weight->is_const = 0;
	weight->quant_channel = 1;
	weight->dtype = CSINN_DTYPE_FLOAT32;
	weight->data = malloc(1 * 65 * 16 * 9 * sizeof(float));
	struct csinn_tensor *bias = csinn_alloc_tensor(sess);
	bias->dim[0] = 1;
	bias->dim[1] = 65;
	bias->dim[2] = 1;
	bias->dim[3] = 1;
	bias->dim_count = 4;
	bias->layout = CSINN_LAYOUT_NCHW;
	bias->is_const = 0;
	bias->quant_channel = 1;
	bias->dtype = CSINN_DTYPE_FLOAT32;
	bias->data = malloc(1 * 65 * 1 * 1 * sizeof(float));
	struct csinn_tensor *output_Conv = csinn_alloc_tensor(sess);
	output_Conv->dim[0] = 4;
	output_Conv->dim[1] = 65;
	output_Conv->dim[2] = 100;
	output_Conv->dim[3] = 100;
	output_Conv->dim_count = 4;
	output_Conv->layout = CSINN_LAYOUT_NCHW;
	output_Conv->is_const = 0;
	output_Conv->quant_channel = 1;
	output_Conv->dtype = CSINN_DTYPE_FLOAT32;
	output_Conv->data = malloc(4 * 65 * 100 * 100 * sizeof(float));
	csinn_conv2d_init(input, output_Conv, weight, bias, line_12_param);
	csinn_conv2d(input, output_Conv, weight, bias, line_12_param);
	printf("Run graph completed.\n");
	return 0;
}
