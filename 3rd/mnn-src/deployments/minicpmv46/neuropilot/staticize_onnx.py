#!/usr/bin/env python3
"""Replace trace-time constant ONNX If/Squeeze branches with static Squeeze ops."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input, load_external_data=False)
    rewritten = []
    replacements = 0

    for node in model.graph.node:
        if node.op_type != "If":
            rewritten.append(node)
            continue

        branches = {attribute.name: attribute.g for attribute in node.attribute}
        then_branch = branches.get("then_branch")
        else_branch = branches.get("else_branch")
        if then_branch is None or else_branch is None:
            raise RuntimeError(f"Malformed If node: {node.name}")

        else_identity = next(
            (branch_node for branch_node in else_branch.node if branch_node.op_type == "Identity"),
            None,
        )
        if else_identity is None or len(else_identity.input) != 1:
            raise RuntimeError(f"Unsupported If branch in {node.name}")
        rewritten.append(
            helper.make_node(
                "Identity",
                inputs=[else_identity.input[0]],
                outputs=list(node.output),
                name=f"{node.name}/StaticIdentity",
            )
        )
        replacements += 1

    if replacements == 0:
        raise RuntimeError("No supported If nodes were found")

    normalized = []
    for node in rewritten:
        if node.op_type == "Gather":
            axis = next(
                (helper.get_attribute_value(attribute) for attribute in node.attribute if attribute.name == "axis"),
                0,
            )
            if axis == 1 and "vit_merger/Gather" in node.name:
                # The two gathers are a fixed 2x2 window permutation and its
                # inverse for the traced 36x28 patch grid. Expressing them as
                # reshapes/transposes keeps the graph on MDLA and avoids the
                # SDK's OpenCL gather compiler path.
                is_inverse = node.name.endswith("Gather_1")
                first_shape = (
                    [1, 18, 14, 2, 2, 1152]
                    if is_inverse
                    else [1, 18, 2, 14, 2, 1152]
                )
                first_shape_name = f"{node.name}/window_shape"
                permuted_name = f"{node.name}/permuted"
                final_shape_name = f"{node.name}/flat_shape"
                normalized.extend(
                    [
                        helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[first_shape_name],
                            name=f"{node.name}/WindowShape",
                            value=numpy_helper.from_array(np.array(first_shape, dtype=np.int64)),
                        ),
                        helper.make_node(
                            "Reshape",
                            inputs=[node.input[0], first_shape_name],
                            outputs=[f"{node.name}/windowed"],
                            name=f"{node.name}/WindowReshape",
                        ),
                        helper.make_node(
                            "Transpose",
                            inputs=[f"{node.name}/windowed"],
                            outputs=[permuted_name],
                            name=f"{node.name}/WindowTranspose",
                            perm=[0, 1, 3, 2, 4, 5],
                        ),
                        helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=[final_shape_name],
                            name=f"{node.name}/FlatShape",
                            value=numpy_helper.from_array(
                                np.array([1, 1008, 1152], dtype=np.int64)
                            ),
                        ),
                        helper.make_node(
                            "Reshape",
                            inputs=[permuted_name, final_shape_name],
                            outputs=list(node.output),
                            name=f"{node.name}/FlatReshape",
                        ),
                    ]
                )
                continue
        if node.op_type == "Slice":
            for input_index in range(1, len(node.input)):
                source_name = node.input[input_index]
                cast_name = f"{node.name}/index_{input_index}_int64"
                normalized.append(
                    helper.make_node(
                        "Cast",
                        inputs=[source_name],
                        outputs=[cast_name],
                        name=f"{node.name}/CastIndex{input_index}",
                        to=onnx.TensorProto.INT64,
                    )
                )
                node.input[input_index] = cast_name
        normalized.append(node)

    del model.graph.node[:]
    model.graph.node.extend(normalized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    onnx.checker.check_model(str(args.output))
    print(f"Replaced {replacements} If node(s): {args.output}")


if __name__ == "__main__":
    main()
