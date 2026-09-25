#!/usr/bin/env python3
"""
CLI: Test MQTT pipeline (mock publisher + subscriber)
Author: Ali Naderi

Usage:
    # Terminal 1: Start subscriber
    python scripts/run_mqtt_test.py --mode subscriber

    # Terminal 2: Start mock publisher @ 3fps
    python scripts/run_mqtt_test.py --mode mock --frames 100

    # For AWS IoT:
    python scripts/run_mqtt_test.py --mode subscriber --endpoint your-endpoint.iot.us-east-1.amazonaws.com --port 8883 --tls --cert-dir ./certs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Test MQTT pipeline")
    parser.add_argument("--mode", required=True, choices=["subscriber", "mock", "both"], help="Mode")
    parser.add_argument("--endpoint", default="localhost", help="MQTT broker endpoint")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="dji/matrice4td/inference")
    parser.add_argument("--fps", type=float, default=3.0, help="For mock publisher")
    parser.add_argument("--frames", type=int, default=50, help="For mock publisher")
    parser.add_argument("--tls", action="store_true", help="Use TLS for AWS")
    parser.add_argument("--cert-dir", default="./certs")
    parser.add_argument("--timeout", type=int, default=None, help="Subscriber timeout seconds")
    
    args = parser.parse_args()
    
    if args.mode == "subscriber":
        from src.mqtt.aws_subscriber import DJIInferenceSubscriber
        
        logger.info("=== Starting MQTT Subscriber ===")
        subscriber = DJIInferenceSubscriber(
            endpoint=args.endpoint,
            port=args.port,
            topic=args.topic,
            use_tls=args.tls,
            cert_dir=args.cert_dir
        )
        subscriber.start(timeout=args.timeout)
    
    elif args.mode == "mock":
        from src.mqtt.dji_mock_publisher import DJIMockPublisher
        
        logger.info("=== Starting Mock Publisher ===")
        publisher = DJIMockPublisher(
            endpoint=args.endpoint,
            port=args.port,
            topic=args.topic,
            fps=args.fps
        )
        publisher.start(num_frames=args.frames)
    
    elif args.mode == "both":
        logger.info("Both mode requires two terminals. Use:")
        logger.info("  Terminal 1: python scripts/run_mqtt_test.py --mode subscriber")
        logger.info("  Terminal 2: python scripts/run_mqtt_test.py --mode mock --frames 100")


if __name__ == "__main__":
    main()
