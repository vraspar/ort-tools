import numpy as np
import onnx
from onnx import TensorProto, helper
import argparse

def create_matmul_chain_model(num_matmuls=100, input_shape=(512, 512), second_input_shape=None, vary_dims=False, data_type=TensorProto.FLOAT):
    """
    Creates an ONNX model with a chain of MatMul operations, with weights as external inputs
    Args:
        num_matmuls: Number of MatMul operations (e.g., 100)
        input_shape: Shape of the first input tensor (e.g., (2, 256, 2048))
        second_input_shape: Shape of the second input tensor (e.g., (2048, 8192)) for the first MatMul
        vary_dims: If True, randomly varies dimensions (keeping compatibility)
        data_type: ONNX data type
    """
    # Validate input shapes
    if second_input_shape:
        if input_shape[-1] != second_input_shape[-2]:
            raise ValueError(f"Incompatible shapes: input_shape[-1] ({input_shape[-1]}) must match second_input_shape[-2] ({second_input_shape[-2]})")

    # Generate dimensions
    dimensions = [input_shape]
    if second_input_shape:
        # First MatMul output shape
        first_output_shape = (*input_shape[:-1], second_input_shape[-1])
        dimensions.append(first_output_shape)
    else:
        # Use default weight shape for the first MatMul
        dimensions.append((*input_shape[:-1], input_shape[-1], input_shape[-1]))

    if vary_dims:
        for _ in range(1, num_matmuls):  # Start from the second MatMul
            m = dimensions[-1][-1]  # Previous last dimension becomes next second-to-last dimension
            n = np.random.randint(64, 1024)  # Random n
            dimensions.append((*dimensions[-1][:-2], m, n))
    else:
        for _ in range(1, num_matmuls):  # Start from the second MatMul
            dimensions.append((*dimensions[-1][:-2], dimensions[-1][-1], input_shape[-1]))

    # Generate model name dynamically
    model_name = f"matmul_chain_{num_matmuls}_matmuls"
    model_name += f"_input_{'x'.join(map(str, input_shape))}"
    if second_input_shape:
        model_name += f"_second_input_{'x'.join(map(str, second_input_shape))}"
    if vary_dims:
        model_name += "_vary_dims"

    if data_type == TensorProto.FLOAT16:
        model_name += "_f16"

    # Generate output path
    output_path = f"{model_name}.onnx"

    # Create input tensor
    inputs = [
        helper.make_tensor_value_info('input', data_type, list(input_shape))
    ]

    # Handle second input for the first MatMul
    if second_input_shape:
        inputs.append(
            helper.make_tensor_value_info('second_input', data_type, list(second_input_shape))
        )
    else:
        inputs.append(
            helper.make_tensor_value_info('weight_0_0', data_type, [input_shape[-1], dimensions[1][-1]])  # Ensure unique name
        )

    # Create weight tensors as inputs for subsequent MatMuls
    for i in range(1 if second_input_shape else 0, num_matmuls):
        inputs.append(
            helper.make_tensor_value_info(
                f'weight_{i}',
                data_type,
                [dimensions[i][-1], dimensions[i + 1][-1]]
            )
        )

    # Create output tensor
    outputs = [
        helper.make_tensor_value_info(
            'output',
            data_type,
            list(dimensions[-1])
        )
    ]

    # Create nodes
    nodes = []
    prev_output = 'input'
    for i in range(num_matmuls):
        input_name = 'second_input' if i == 0 and second_input_shape else (f'weight_0_0' if i == 0 else f'weight_{i}')
        output_name = f'matmul_out_{i}' if i < num_matmuls - 1 else 'output'
        nodes.append(
            helper.make_node(
                'MatMul',
                inputs=[prev_output, input_name],
                outputs=[output_name],
                name=f'MatMul_{i}'
            )
        )
        prev_output = output_name

    # Create graph
    graph = helper.make_graph(
        nodes,
        model_name,
        inputs,
        outputs,
        initializer=[]  # No initializers since weights are external inputs
    )

    model = helper.make_model(
        graph,
        producer_name='xAI',
        opset_imports=[helper.make_opsetid('', 13)]
    )

    # Check and save model
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Model saved: {output_path}")
    print(f"Model name: {model_name}")
    print(f"Inputs: {len(inputs)} (input + {num_matmuls} weights{' + second input' if second_input_shape else ''})")
    print(f"Outputs: 1")
    print(f"Nodes: {len(nodes)} MatMul operations")
    if vary_dims:
        print(f"Dimensions: {dimensions}")
    else:
        print(f"Input shape: {dimensions[0]}, Weight shapes: {dimensions[1]}, Output shape: {dimensions[-1]}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create an ONNX model with a chain of MatMul operations.")
    parser.add_argument("--num_matmuls", type=int, default=100, help="Number of MatMul operations (default: 100)")
    parser.add_argument("--input_shape", type=int, nargs='+', default=[512, 512], help="Shape of the first input tensor (e.g., 2 256 2048) (default: 512x512)")
    parser.add_argument("--second_input_shape", type=int, nargs='+', help="Shape of the second input tensor (e.g., 2048 8192) for the first MatMul (default: None)")
    parser.add_argument("--vary_dims", action="store_true", help="Randomly vary dimensions (default: False)")
    parser.add_argument("--data_type", type=str, choices=["float32", "float16"], default="float32", help="Data type for the tensors (default: float32)")

    args = parser.parse_args()

    data_type = TensorProto.FLOAT if args.data_type == "float32" else TensorProto.FLOAT16

    # Create model
    create_matmul_chain_model(
        num_matmuls=args.num_matmuls,
        input_shape=tuple(args.input_shape),
        second_input_shape=tuple(args.second_input_shape) if args.second_input_shape else None,
        vary_dims=args.vary_dims,
        data_type=data_type
    )
