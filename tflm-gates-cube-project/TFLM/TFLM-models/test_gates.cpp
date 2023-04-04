/**
 ******************************************************************************
 * @file         : test_gates.cpp
 * @brief        : Test neural networks, trained to represent logical gates.
 * @author		 : paspf
 * @date		 : 2023-04-01
 * @copyright	 : paspf, GNU General Public License v3.0
 *
 * @see https://www.tensorflow.org/lite/microcontrollers/get_started_low_level#run_inference
 ******************************************************************************
 */

#include "test_gates.hpp"
#include <math.h>
#include <cstdio>

/* TFLM includes */
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Model includes */
extern unsigned int model_and_gate_len;
extern unsigned char model_and_gate[];
#include "nor-gate.h"
#include "or-gate.h"
#include "xnor-gate.h"
#include "xor-gate.h"

/* Test data */
const uint32_t len_input_data = 4;
const float input_data[4][2] = { { 0.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 0.0f },
		{ 1.0f, 1.0f } };

/* Reference output data */
const int output_data_and_gate[] = { 0, 0, 0, 1 };
const int output_data_nor_gate[] = { 1, 1, 1, 0 };
const int output_data_or_gate[] = { 0, 1, 1, 1 };
const int output_data_xor_gate[] = { 0, 1, 1, 0 };
const int output_data_xnor_gate[] = { 1, 0, 0, 1 };

/* Private functions */
static int _test_gate(const void *tflm_model, const float golden_inputs[4][2],
		const int *golden_outputs);

/* Register all operations required for the networks */
namespace {
using GatesOpsResolver = tflite::MicroMutableOpResolver<2>;

TfLiteStatus RegisterOps(GatesOpsResolver &op_resolver) {
	TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
	TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
	return kTfLiteOk;
}
}  // namespace.

/* Run the tests */
void test_gates() {
	tflite::InitializeTarget();
	test_and_gate();
	test_nor_gate();
	test_or_gate();
	test_xnor_gate();
	test_xor_gate();
}

int test_and_gate() {
	return _test_gate(model_and_gate, input_data, output_data_and_gate);
}

int test_nor_gate() {
	printf("test_nand_gate_tflm_8bit\n");
	return _test_gate(model_nor_gate, input_data, output_data_nor_gate);
}
int test_or_gate() {
	printf("test_or_gate_tflm_8bit\n");
	return _test_gate(model_or_gate, input_data, output_data_or_gate);
}
int test_xnor_gate() {
	printf("test_xnor_gate_tflm_8bit\n");
	return _test_gate(model_xnor_gate, input_data, output_data_xnor_gate);
}

int test_xor_gate() {
	printf("test_xor_gate_tflm_8bit\n");
	return _test_gate(model_xor_gate, input_data, output_data_xor_gate);
}

/**
 * @fn int _test_gate(const void*, const float[][], const int*)
 * @brief Test the TFLM model of a gate.
 *
 * @param tflm_model TFLM model to test.
 * @param golden_inputs Reference inputs.
 * @param golden_outputs Reference outputs.
 * @return kTfLiteOk on success.
 */
static int _test_gate(const void *tflm_model, const float golden_inputs[4][2],
		const int *golden_outputs) {
	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	const tflite::Model *model = ::tflite::GetModel(tflm_model);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		MicroPrintf("Model provided is schema version %d not equal "
				"to supported version %d.\n", model->version(),
				TFLITE_SCHEMA_VERSION);
	}

	GatesOpsResolver op_resolver;
	TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

	// Arena size just a round number. The exact arena usage can be determined
	// using the RecordingMicroInterpreter.
	constexpr int kTensorArenaSize = 2056;
	uint8_t tensor_arena[kTensorArenaSize];

	// Build an interpreter to run the model with
	tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
			kTensorArenaSize);

	// Allocate memory from the tensor_arena for the model's tensors
	if (interpreter.AllocateTensors() != kTfLiteOk) {
		MicroPrintf("Allocate tensor failed.");
		return kTfLiteError;
	}

	// Obtain a pointer to the model's input tensor.
	TfLiteTensor *input = interpreter.input(0);

	// Make sure the input has the properties we expect.
	if (input == nullptr) {
		MicroPrintf("Input tensor is null.");
		return kTfLiteError;
	}
	// Get the input quantization parameters.
	float input_scale = input->params.scale;
	int input_zero_point = input->params.zero_point;

	// Obtain a pointer to the output tensor.
	TfLiteTensor *output = interpreter.output(0);

	// Get the output quantization parameters.
	float output_scale = output->params.scale;
	int output_zero_point = output->params.zero_point;

	for (uint32_t i = 0; i < len_input_data; ++i) {
		// Fill inputs of neural network.
		input->data.int8[0] = float_to_quant(golden_inputs[i][0], input_scale,
				input_zero_point);
		input->data.int8[1] = float_to_quant(golden_inputs[i][1], input_scale,
				input_zero_point);
		// Run prediction.
		interpreter.Invoke();
		// Get outputs of neural network.
		float y_pred = quant_to_float(output->data.int8[0], output_scale,
				output_zero_point);
		// Print results.
		print_single_prediction(golden_inputs[i], golden_outputs[i], y_pred,
				float_prediction_to_binary_int(y_pred));
	}
	return kTfLiteOk;
}

/**
 * @fn int float_prediction_to_binary_int(float)
 * @brief Convert float value to binary int value (0 or 1).
 * If prediction > 0.5 -> return 1
 * Else -> return 0
 *
 * @param prediction Float value to convert.
 * @return Int value, 0 or 1.
 */
int float_prediction_to_binary_int(const float prediction) {
	if (prediction > 0.5) {
		return 1;
	}
	return 0;
}

/**
 * @fn void print_single_prediction(float*, int, int)
 * @brief Print out a single prediction.
 *
 * @param x_input x-network input.
 * @param y_ref   y-network reference.
 * @param y_pred_float  y-network output as float.
 * @param y_pred_int y-network output as int.
 * @return 1 if y_pred matches y_ref.
 */
int print_single_prediction(const float *x_input, const int y_ref,
		const float y_pred_float, const int y_pred_int) {
	if (y_ref == y_pred_int) {
		printf("|[%.1f, %.1f]|[%d]| -> %f -> Pass\n", x_input[0], x_input[1],
				y_ref, y_pred_float);
		return 1;
	}
	printf("|[%.1f, %.1f]|[%d]| -> %f -> Fail\n", x_input[0], x_input[1], y_ref,
			y_pred_float);
	return 0;
}

/**
 * @fn int8_t float_to_quant(float, float, int)
 * @brief Quantize float value.
 *
 * @param input Float input.
 * @param scale Quantization scale.
 * @param zero_point Quantization zero offset.
 * @return Quantized value.
 */
int8_t float_to_quant(float input, float scale, int zero_point) {
	return (int8_t) (input / scale + zero_point);
}

/**
 * @fn float quant_to_float(int32_t, float, int)
 * @brief Bring quantized value back to float.
 *
 * @param quant Quantized value.
 * @param scale Quantization scale.
 * @param zero_point Quantization zero offset.
 * @return Float value.
 */
float quant_to_float(int32_t quant, float scale, int zero_point) {
	return (quant - zero_point) * scale;
}

