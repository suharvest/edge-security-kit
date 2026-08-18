"""CLI entry point: ``python -m esk_jetson --config config.yaml``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import Config
from .detector import Detector, install_signal_handlers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esk_jetson", description=__doc__)
    parser.add_argument("--config", help="YAML or JSON config file")
    parser.add_argument("--source", help="override config.source")
    parser.add_argument("--device-id", help="override config.device_id")
    parser.add_argument("--stream-id", help="override config.stream_id")
    parser.add_argument("--model", help="override config.model")
    parser.add_argument("--mqtt-host", help="override config.mqtt_host")
    parser.add_argument("--mqtt-port", type=int, help="override config.mqtt_port")
    parser.add_argument("--preview-port", type=int, help="override config.preview_port")
    parser.add_argument(
        "--allow-sw-decode",
        action="store_true",
        help="permit the CPU decoder fallback (reported as decode: sw)",
    )
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
        "--stages",
        default="",
        help="write the per-stage p50/p95 timing breakdown to this JSON path on exit",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="load the engine and the config, then exit without connecting",
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
    if args.allow_sw_decode:
        cfg.require_hw_decode = False

    if args.validate:
        from .trt_yolo import TRTPersonDetector

        model = TRTPersonDetector(
            cfg.model,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=cfg.input_size,
        )
        print(
            f"engine OK: {cfg.model} backend={model.backend} "
            f"input={model.input_dtype} output={model.output_shape} {model.output_dtype}"
        )
        model.close()
        return 0

    detector = Detector(cfg)
    install_signal_handlers(detector)
    detector.run(max_frames=args.frames, max_seconds=args.seconds, annotate=args.annotate)
    if args.stages:
        summary = detector.stages.summary()
        with open(args.stages, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
