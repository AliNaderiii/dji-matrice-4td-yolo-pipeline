"""
DJI Mock Publisher - Simulates Matrice 4TD + Dock 3 publishing inference JSON @ 3fps
Author: Ali Naderi

For local testing without real drone. Publishes realistic payloads to MQTT
so subscriber can be tested and validated for deliverable.


"""

import json
import time
import random
from pathlib import Path
from typing import List
from loguru import logger
import paho.mqtt.client as mqtt
from datetime import datetime


class DJIMockPublisher:
    """
    Simulates DJI Matrice 4TD publishing inference results.
    
    Publishes JSON payloads at ~3fps to MQTT topic, mimicking real NPU output.
    Useful for:
    - Testing subscriber without drone
    - Validating AWS IoT integration
    - Demo for validation
    - CI/CD testing
    """

    def __init__(self, 
                 endpoint: str = "localhost",
                 port: int = 1883,
                 topic: str = "dji/matrice4td/inference",
                 fps: float = 3.0,
                 drone_sn: str = "4TD-SIM-001"):
        
        self.endpoint = endpoint
        self.port = port
        self.topic = topic
        self.fps = fps
        self.drone_sn = drone_sn
        self.interval = 1.0 / fps
        
        self.client = mqtt.Client(client_id=f"dji_mock_{int(time.time())}")
        self.client.on_connect = self._on_connect
        
        self.frame_id = 0
        self.running = False
        
        logger.info(f"Mock Publisher: {endpoint}:{port}, topic={topic}, {fps}fps")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.success(f"Mock publisher connected to {self.endpoint}:{self.port}")
        else:
            logger.error(f"Mock publisher connection failed: {rc}")

    def _generate_realistic_detections(self) -> List[dict]:
        """Generate realistic detections for construction site"""
        detections = []
        
        # Random number of objects (0-5)
        num_persons = random.randint(0, 3)
        num_vehicles = random.randint(0, 2)
        
        for _ in range(num_persons):
            # Person
            x_min = random.uniform(50, 500)
            y_min = random.uniform(50, 400)
            w = random.uniform(30, 80)
            h = random.uniform(80, 200)
            
            has_hardhat = random.random() > 0.3  # 70% have hard-hat, 30% violation
            
            # Person bbox
            detections.append({
                "class_id": 0,
                "class_name": "Person",
                "confidence": round(random.uniform(0.75, 0.95), 3),
                "bbox": [round(x_min, 1), round(y_min, 1), round(x_min+w, 1), round(y_min+h, 1)]
            })
            
            # Hard-hat or No-Hard-Hat (slightly smaller bbox on head)
            head_x = x_min + w*0.2
            head_y = y_min
            head_w = w*0.6
            head_h = h*0.2
            
            if has_hardhat:
                detections.append({
                    "class_id": 2,
                    "class_name": "Hard-Hat",
                    "confidence": round(random.uniform(0.8, 0.96), 3),
                    "bbox": [round(head_x, 1), round(head_y, 1), round(head_x+head_w, 1), round(head_y+head_h, 1)]
                })
            else:
                detections.append({
                    "class_id": 3,
                    "class_name": "No-Hard-Hat",
                    "confidence": round(random.uniform(0.7, 0.92), 3),
                    "bbox": [round(head_x, 1), round(head_y, 1), round(head_x+head_w, 1), round(head_y+head_h, 1)]
                })
        
        for _ in range(num_vehicles):
            x_min = random.uniform(100, 400)
            y_min = random.uniform(200, 500)
            w = random.uniform(80, 200)
            h = random.uniform(40, 100)
            
            detections.append({
                "class_id": 1,
                "class_name": "Vehicle",
                "confidence": round(random.uniform(0.8, 0.95), 3),
                "bbox": [round(x_min, 1), round(y_min, 1), round(x_min+w, 1), round(y_min+h, 1)]
            })
        
        return detections

    def _generate_payload(self) -> dict:
        """Generate full payload"""
        self.frame_id += 1
        
        payload = {
            "timestamp": int(time.time() * 1000),
            "drone_sn": self.drone_sn,
            "frame_id": self.frame_id,
            "detections": self._generate_realistic_detections(),
            "inference_time_ms": round(random.uniform(35, 55), 1),
            "model_version": "yolov8n_dji_4class_v1_INT8",
            # DJI Cloud API compatible fields
            "bid": f"bid-{self.frame_id}",
            "tid": f"tid-{int(time.time())}",
            "method": "ai_inference_result"
        }
        
        return payload

    def start(self, num_frames: int = 100):
        """
        Start publishing
        
        Args:
            num_frames: Number of frames to publish (None = infinite)
        """
        logger.info(f"🚀 Starting mock publisher: {num_frames} frames @ {self.fps}fps")
        logger.info(f"   Topic: {self.topic}")
        logger.info(f"   Drone SN: {self.drone_sn}")
        
        try:
            self.client.connect(self.endpoint, self.port, 60)
            self.client.loop_start()
            
            self.running = True
            published = 0
            
            while self.running and (num_frames is None or published < num_frames):
                payload = self._generate_payload()
                payload_str = json.dumps(payload)
                
                result = self.client.publish(self.topic, payload_str)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Published frame {self.frame_id}: {len(payload['detections'])} detections "
                                f"(P:{len([d for d in payload['detections'] if d['class_name']=='Person'])} "
                                f"NHH:{len([d for d in payload['detections'] if d['class_name']=='No-Hard-Hat'])})")
                else:
                    logger.error(f"Publish failed: {result.rc}")
                
                published += 1
                time.sleep(self.interval)
            
            logger.success(f"Published {published} frames")
            
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        except Exception as e:
            logger.error(f"Publisher error: {e}")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DJI Mock Publisher @ 3fps")
    parser.add_argument("--endpoint", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--topic", default="dji/matrice4td/inference")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--frames", type=int, default=50)
    args = parser.parse_args()
    
    publisher = DJIMockPublisher(
        endpoint=args.endpoint,
        port=args.port,
        topic=args.topic,
        fps=args.fps
    )
    publisher.start(num_frames=args.frames)
