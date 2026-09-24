#!/usr/bin/env python3
"""Run MTK Converter on an ONNX model that stores weights as external data.

NeuroPilot Premium 9.0.9 copies large models to a temporary directory during
shape inference but does not copy their external tensor files.  Infer shapes
beside the source graph, where those relative paths remain valid, then use the
normal MTK conversion pipeline without repeating that broken copy step.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import shape_inference

from mtk_converter.python.converters import base_converter
from mtk_converter.python.converters.onnx import converter as onnx_converter


def _use_preinferred_model(self: object, model: onnx.ModelProto) -> onnx.ModelProto:
    del self
    onnx.helper.strip_doc_string(model)
    return model


def main() -> None:
    parser = onnx_converter.get_argument_parser()
    args = base_converter.parse_args(parser)

    source_path = Path(args.input_model_file).resolve()
    inferred_path = source_path.with_name(f"{source_path.stem}.mtk-inferred.onnx")

    print(f"Inferring shapes beside external weights: {source_path}", flush=True)
    shape_inference.infer_shapes_path(str(source_path), str(inferred_path))
    onnx.checker.check_model(str(inferred_path))

    # Shape inference is already complete. Avoid the SDK's second pass, which
    # relocates only the ONNX protobuf and loses its external tensor files.
    onnx_converter.OnnxConverter._clone_and_polish_model = _use_preinferred_model

    try:
        converter = onnx_converter.OnnxConverter.from_model_proto_file(
            str(inferred_path), args.input_names, args.input_shapes, args.output_names
        )
        converter._set_options_from_argparse(args)

        if args.output_file_format == "tflite":
            converter.convert_to_tflite(
                output_file=args.output_file,
                tflite_op_export_spec=args.tflite_op_export_spec,
                custom_description=args.tflite_custom_description,
            )
        else:
            converter.convert_to_mlir(
                output_file=args.output_file,
                preserve_tensor_names=args.mlir_preserve_tensor_names,
                mlir_export_spec=args.mlir_export_spec,
            )
    finally:
        inferred_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
