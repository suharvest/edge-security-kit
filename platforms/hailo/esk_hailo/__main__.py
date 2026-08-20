"""CLI entry point: ``uv run python -m esk_hailo --config config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Config
from .detector import Detector, install_signal_handlers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esk_hailo", description=__doc__)
    parser.add_argument("--config", help="YAML or JSON config file")
    parser.add_argument("--source", help="override config.source")
    parser.add_argument("--device-id", help="override config.device_id")
    parser.add_argument("--stream-id", help="override config.stream_id")
    parser.add_argument("--model", help="override config.model")
    parser.add_argument("--mqtt-host", help="override config.mqtt_host")
    parser.add_argument("--mqtt-port", type=int, help="override config.mqtt_port")
    parser.add_argument("--preview-port", type=int, help="override config.preview_port")
    parser.add_argument(
        "--frames", type=int, default=0, help="exit after N published frames (0 = forever)"
    )
    parser.add_argument(
        "--seconds", type=float, default=0.0, help="exit after N seconds (0 = forever)"
    )
    parser.add_argument(
        "--annotate", default="", help="write one annotated evidence JPEG to this path"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="configure the HEF on the NPU and print its shapes, then exit",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = Config.load(args.config)
    for name in (
        "source", "device_id", "stream_id", "model", "mqtt_host", "mqtt_port", "preview_port"
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if args.validate:
        from .hailo_yolo import HailoPersonDetector

        model = HailoPersonDetector(
            cfg.model,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=cfg.input_size,
            timeout_ms=cfg.infer_timeout_ms,
        )
        print(f"model OK: {cfg.model} backend={model.backend}")
        print(f"host optimizations: {model.optimizations or 'none'}")
        if model.quants is not None:
            for name, (scale, zero_point) in zip(model.output_names, model.quants):
                print(f"quant {name} qp_scale={scale!r} qp_zp={zero_point!r}")
        print(f"input {model.input_name} {model.input_shape}")
        for name, shape in model.output_shapes.items():
            print(f"output {name} {shape}")
        model.close()
        return 0

    detector = Detector(cfg)
    install_signal_handlers(detector)
    detector.run(max_frames=args.frames, max_seconds=args.seconds, annotate=args.annotate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
