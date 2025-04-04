import numpy as np
import onnx
from onnx import TensorProto, helper
import argparse  # Added for command-line argument parsing

def create_matmul_chain_model(num_matmuls=100, base_dims=(512, 512), vary_dims=False, data_type=TensorProto.FLOAT, output_path="matmul_chain.onnx", second_input=None):
    """
    Creates an ONNX model with a chain of MatMul operations, with weights as external inputs
    Args:
        num_matmuls: Number of MatMul operations (e.g., 100)
        base_dims: Tuple (m,n) for base matrix dimensions
        vary_dims: If True, randomly varies dimensions (keeping compatibility)
        data_type: ONNX data type
        output_path: Path to save the ONNX model
        second_input: Optional second input tensor for the first MatMul
    """
    # Generate dimensions
    dimensions = [base_dims]
    if vary_dims:
        for _ in range(num_matmuls):
            m = dimensions[-1][1]  # Previous n becomes next m
            n = np.random.randint(64, 1024)  # Random n
            dimensions.append((m, n))
    else:
        for _ in range(num_matmuls):
            dimensions.append((dimensions[-1][1], base_dims[1]))

    # Create input tensor
    inputs = [
        helper.make_tensor_value_info('input', data_type, [dimensions[0][0], dimensions[0][1]])
    ]

    # Handle second input for the first MatMul
    if second_input:
        inputs.append(
            helper.make_tensor_value_info('second_input', data_type, [dimensions[0][1], dimensions[1][1]])
        )
    else:
        inputs.append(
            helper.make_tensor_value_info('weight_0', data_type, [dimensions[0][1], dimensions[1][1]])
        )

    # Create weight tensors as inputs for subsequent MatMuls
    for i in range(1 if second_input else 0, num_matmuls):
        inputs.append(
            helper.make_tensor_value_info(
                f'weight_{i}',
                data_type,
                [dimensions[i][1], dimensions[i + 1][1]]
            )
        )

    # Create output tensor
    outputs = [
        helper.make_tensor_value_info(
            'output',
            data_type,
            [dimensions[0][0], dimensions[-1][1]]
        )
    ]

    # Create nodes
    nodes = []
    prev_output = 'input'
    for i in range(num_matmuls):
        input_name = 'second_input' if i == 0 and second_input else f'weight_{i}'
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
        'matmul_chain',
        inputs,
        outputs,
        initializer=[]  # No initializers since weights are external inputs
    )

    # Create model
    model = helper.make_model(
        graph,
        producer_name='xAI',
        opset_imports=[helper.make_opsetid('', 13)]
    )

    # Check and save model
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Model saved: {output_path}")
    print(f"Inputs: {len(inputs)} (input + {num_matmuls} weights)")
    print(f"Outputs: 1")
    print(f"Nodes: {len(nodes)} MatMul operations")
    if vary_dims:
        print(f"Dimensions: {dimensions}")
    else:
        print(f"Input shape: {dimensions[0]}, Weight shapes: {dimensions[1]}, Output shape: {dimensions[0][0], dimensions[-1][1]}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create an ONNX model with a chain of MatMul operations.")
    parser.add_argument("--num_matmuls", type=int, default=100, help="Number of MatMul operations (default: 100)")
    parser.add_argument("--base_dims", type=int, nargs=2, default=(512, 512), help="Base dimensions (m, n) for matrices (default: 512x512)")
    parser.add_argument("--vary_dims", action="store_true", help="Randomly vary dimensions (default: False)")
    parser.add_argument("--output_path", type=str, default="matmul_chain.onnx", help="Output path for the ONNX model (default: matmul_chain.onnx)")
    parser.add_argument("--second_input", action="store_true", help="Use a second input tensor for the first MatMul (default: False)")

    args = parser.parse_args()

    # Create model
    create_matmul_chain_model(
        num_matmuls=args.num_matmuls,
        base_dims=tuple(args.base_dims),
        vary_dims=args.vary_dims,
        output_path=args.output_path,
        second_input=args.second_input
    )
