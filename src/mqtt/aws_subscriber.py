"""
DJI Inference Subscriber - AWS IoT Core & Local EMQX
Author: Ali Naderi | Edge AI

Subscribes to MQTT topic where Matrice 4TD publishes inference JSON @ 3fps
Handles TLS, reconnection, payload validation, FPS monitoring, safety alerts.

Advanced: Includes Pydantic validation, alerting, logging, AWS integration.
"""

import json
import time
import ssl
from pathlib import Path
from typing import Optional, Callable
from loguru import logger
import paho.mqtt.client as mqtt
from datetime import datetime

from .payload_models import InferencePayload, AlertPayload


class DJIInferenceSubscriber:
    """
    Production MQTT subscriber for DJI Matrice 4TD inference results.
    
    Features:
    - TLS support for AWS IoT Core
    - Automatic reconnection
    - Pydantic payload validation
    - FPS monitoring (should be ~3fps)
    - Safety violation alerts
    - JSONL logging for validation proof
    - Pluggable callbacks
    """

    def __init__(self, 
                 endpoint: str = "localhost",
                 port: int = 1883,
                 topic: str = "dji/matrice4td/inference",
                 client_id: Optional[str] = None,
                 use_tls: bool = False,
                 cert_dir: Optional[str] = None,
                 log_file: str = "demo/dji_inference_log.jsonl",
                 on_detection_callback: Optional[Callable] = None):
        
        self.endpoint = endpoint
        self.port = port
        self.topic = topic
        self.use_tls = use_tls
        self.cert_dir = Path(cert_dir) if cert_dir else None
        self.log_file = Path(log_file)
        self.on_detection_callback = on_detection_callback
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.total_detections = 0
        self.safety_violations = 0
        
        # MQTT client
        client_id = client_id or f"dji_sub_{int(time.time())}"
        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # TLS setup for AWS
        if self.use_tls:
            self._setup_tls()
        
        # Ensure log dir exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Subscriber initialized: {endpoint}:{port}, topic={topic}, TLS={use_tls}")

    def _setup_tls(self):
        """Setup TLS for AWS IoT Core"""
        if not self.cert_dir or not self.cert_dir.exists():
            logger.warning(f"Cert dir {self.cert_dir} not found, TLS may fail. Provide AmazonRootCA1.pem, certificate.pem.crt, private.pem.key")
            # Try default locations
            self.cert_dir = Path("./certs")
        
        try:
            ca_path = self.cert_dir / "AmazonRootCA1.pem"
            cert_path = self.cert_dir / "certificate.pem.crt"
            key_path = self.cert_dir / "private.pem.key"
            
            if ca_path.exists() and cert_path.exists() and key_path.exists():
                self.client.tls_set(
                    ca_certs=str(ca_path),
                    certfile=str(cert_path),
                    keyfile=str(key_path),
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLSv1_2
                )
                logger.info(f"TLS configured with certs from {self.cert_dir}")
            else:
                logger.warning(f"Certs not found in {self.cert_dir}, using default TLS (may fail for AWS)")
                self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2)
        except Exception as e:
            logger.error(f"TLS setup failed: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected"""
        if rc == 0:
            logger.success(f"Connected to MQTT broker: {self.endpoint}:{self.port}")
            client.subscribe(self.topic)
            logger.info(f"Subscribed to topic: {self.topic}")
            
            # Also subscribe to wildcard for debugging
            if "dji/" in self.topic:
                client.subscribe("dji/+/inference")
                client.subscribe("dji/#")
        else:
            logger.error(f"Connection failed, rc={rc}: {mqtt.connack_string(rc)}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected"""
        if rc != 0:
            logger.warning(f"Unexpected disconnection, rc={rc}. Auto-reconnecting...")
        else:
            logger.info("Disconnected")

    def _on_message(self, client, userdata, msg):
        """Callback when message received"""
        self.frame_count += 1
        
        try:
            payload_str = msg.payload.decode('utf-8')
            payload_dict = json.loads(payload_str)
            
            # Try to parse as InferencePayload (handles both DJI Cloud API and custom formats)
            inference = self._parse_payload(payload_dict)
            
            if inference:
                self.total_detections += len(inference.detections)
                violations = inference.get_safety_violations()
                self.safety_violations += len(violations)
                
                # Log summary
                summary = inference.to_summary()
                logger.info(f"Frame {self.frame_count} | {summary['total_detections']} objs | "
                            f"P:{summary['persons']} V:{summary['vehicles']} "
                            f"HH:{summary['hard_hats']} NHH:{summary['no_hard_hats']} | "
                            f"FPS: {self._calculate_fps():.1f}")
                
                if violations:
                    logger.warning(f"⚠️ SAFETY VIOLATION: {len(violations)} person(s) without hard-hat! "
                                   f"Conf: {[f'{v.confidence:.2f}' for v in violations]}")
                    
                    # Create alert
                    alert = AlertPayload.from_inference(inference)
                    if alert:
                        logger.warning(f"ALERT: {alert.message}")
                        # Here you could publish to alerts topic, send to Lambda, etc.
                        self._handle_alert(alert)
                
                # Save to log file (for deliverable validation)
                self._save_log(payload_dict)
                
                # Custom callback
                if self.on_detection_callback:
                    self.on_detection_callback(inference)
            else:
                # Raw message (e.g., OSD telemetry)
                logger.debug(f"Raw message on {msg.topic}: {payload_str[:200]}...")
                
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON message on {msg.topic}: {msg.payload[:100]}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.debug(f"Payload: {msg.payload[:500]}")

    def _parse_payload(self, payload_dict: dict) -> Optional[InferencePayload]:
        """Parse various DJI payload formats into unified InferencePayload"""
        try:
            # Format 1: Our custom format (direct InferencePayload)
            if 'detections' in payload_dict and 'timestamp' in payload_dict:
                return InferencePayload(**payload_dict)
            
            # Format 2: DJI Cloud API format: {"data": {"detections": [...]}, "timestamp": ...}
            if 'data' in payload_dict and isinstance(payload_dict['data'], dict):
                data = payload_dict['data']
                if 'detections' in data:
                    # Merge top-level and data
                    merged = {
                        'timestamp': payload_dict.get('timestamp', int(time.time()*1000)),
                        'detections': data['detections'],
                        'drone_sn': payload_dict.get('gateway') or data.get('drone_sn'),
                        'frame_id': data.get('frame_id'),
                        'inference_time_ms': data.get('inference_time_ms'),
                        'model_version': data.get('model_version'),
                        'bid': payload_dict.get('bid'),
                        'tid': payload_dict.get('tid'),
                        'method': payload_dict.get('method')
                    }
                    return InferencePayload(**merged)
            
            # Format 3: Direct list of detections (simplified)
            if isinstance(payload_dict, list):
                return InferencePayload(
                    timestamp=int(time.time()*1000),
                    detections=payload_dict
                )
            
            return None
        except Exception as e:
            logger.debug(f"Payload parsing failed, treating as raw: {e}")
            return None

    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        now = time.time()
        elapsed = now - self.last_fps_time
        
        if elapsed >= 1.0:
            fps = self.frame_count / (now - self.start_time) if (now - self.start_time) > 0 else 0
            # For instant FPS, use frame_count since last check
            # Simplified: return total frames / total time
            return fps
        return 0.0

    def _save_log(self, payload: dict):
        """Append to JSONL log file for validation proof"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(payload) + "\n")
        except Exception as e:
            logger.error(f"Failed to save log: {e}")

    def _handle_alert(self, alert: AlertPayload):
        """Handle safety alert - advanced: can extend to SNS, S3, etc."""
        alert_file = self.log_file.parent / "safety_alerts.jsonl"
        try:
            with open(alert_file, 'a') as f:
                f.write(alert.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")

    def start(self, timeout: Optional[int] = None):
        """
        Start subscriber (blocking)
        
        Args:
            timeout: Optional timeout in seconds (None = forever)
        """
        logger.info(f"🚀 Starting DJI Inference Subscriber")
        logger.info(f"   Broker: {self.endpoint}:{self.port}")
        logger.info(f"   Topic: {self.topic}")
        logger.info(f"   Log: {self.log_file}")
        logger.info(f"   Expected FPS: ~3 (as per specification (3fps))")
        logger.info(f"   Press Ctrl+C to stop\n")
        
        try:
            self.client.connect(self.endpoint, self.port, keepalive=60)
            
            if timeout:
                self.client.loop_start()
                time.sleep(timeout)
                self.client.loop_stop()
                self.client.disconnect()
                logger.info(f"Stopped after {timeout}s timeout")
            else:
                self.client.loop_forever()
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped by user")
            self.client.disconnect()
            self._print_final_stats()
        except Exception as e:
            logger.error(f"Connection error: {e}")
            logger.info("💡 Tips:")
            logger.info("  - For local test: docker-compose -f docker/docker-compose.yml up -d")
            logger.info("  - For AWS: Check endpoint, certs in ./certs/, and IoT policy")
            self._print_final_stats()

    def _print_final_stats(self):
        """Print final statistics"""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        logger.info("\n=== Final Stats ===")
        logger.info(f"Total frames: {self.frame_count}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Avg FPS: {avg_fps:.2f} (target 3.0)")
        logger.info(f"Total detections: {self.total_detections}")
        logger.info(f"Safety violations: {self.safety_violations}")
        logger.info(f"Log file: {self.log_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DJI Matrice 4TD MQTT Subscriber")
    parser.add_argument("--endpoint", default="localhost", help="MQTT broker endpoint")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="dji/matrice4td/inference")
    parser.add_argument("--tls", action="store_true", help="Use TLS for AWS IoT")
    parser.add_argument("--cert-dir", default="./certs")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()
    
    subscriber = DJIInferenceSubscriber(
        endpoint=args.endpoint,
        port=args.port,
        topic=args.topic,
        use_tls=args.tls,
        cert_dir=args.cert_dir
    )
    subscriber.start(timeout=args.timeout)
