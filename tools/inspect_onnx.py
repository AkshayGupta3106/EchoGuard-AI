"""Inspect supplied ONNX files for graph I/O, operators, and weight dtypes."""
import sys
import os
import onnx
from collections import Counter

def inspect_model(path):
    print("=" * 70)
    print(f"FILE: {path}")

    if not os.path.exists(path):
        print("ERROR: File does not exist")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Size: {size_mb:.3f} MB")

    model = onnx.load(path)

    print("\nInputs:")
    for x in model.graph.input:
        print(f"  {x.name}: {x.type.tensor_type.elem_type}")

    print("\nOutputs:")
    for x in model.graph.output:
        print(f"  {x.name}: {x.type.tensor_type.elem_type}")

    print(f"\nTotal nodes: {len(model.graph.node)}")

    op_counts = Counter(node.op_type for node in model.graph.node)
    print(f"Op type counts: {dict(op_counts)}")

    quant_ops = {
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "MatMulInteger",
        "ConvInteger",
        "QLinearMatMul",
        "QLinearConv"
    }

    found_quant = {
        op: count
        for op, count in op_counts.items()
        if op in quant_ops
    }

    if found_quant:
        print(f"\nQUANTIZATION DETECTED: {found_quant}")
    else:
        print("\nNO quantization ops found")

    dtype_counts = Counter()

    total_params = 0

    for initializer in model.graph.initializer:
        dtype_counts[str(initializer.data_type)] += 1

        # Number of elements in tensor
        n = 1
        for d in initializer.dims:
            n *= d

        total_params += n

    print(f"\nInitializer dtypes: {dict(dtype_counts)}")
    print(f"Approx total parameters: {total_params:,}")

    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_onnx.py file1.onnx file2.onnx ...")
        return 1
    for file in sys.argv[1:]:
        inspect_model(file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
