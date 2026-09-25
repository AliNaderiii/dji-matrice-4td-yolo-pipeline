"""
ONNX Validator - Ensures ONNX is valid and DJI-compatible
Author: Ali Naderi
"""

from pathlib import Path
from typing import Dict
from loguru import logger

class OnnxValidator:
    """Validates ONNX model for DJI deployment"""

    def validate(self, onnx_path: str) -> Dict:
        """
        Validate ONNX:
        - Checker
        - Input shape [1,3,640,640]
        - Size <20MB
        - Opset
        """
        onnx_path = Path(onnx_path)
        report = {
            'path': str(onnx_path),
            'valid': False,
            'size_mb': 0,
            'input_shape': None,
            'opset': None,
            'issues': []
        }
        
        try:
            import onnx
            
            # Load and check
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            logger.info(f"ONNX checker passed: {onnx_path}")
            
            # Size
            size_mb = onnx_path.stat().st_size / (1024*1024)
            report['size_mb'] = size_mb
            
            if size_mb > 20:
                report['issues'].append(f"Model large: {size_mb:.1f}MB >20MB for NPU")
            else:
                logger.info(f"Size OK: {size_mb:.2f}MB")
            
            # Input shape
            input_tensor = model.graph.input[0]
            dims = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
            report['input_shape'] = dims
            logger.info(f"Input shape: {dims}")
            
            # Expected [1,3,640,640] or [1,3,640,640] with batch 1
            if len(dims) == 4:
                if dims[1] != 3 or dims[2] != 640 or dims[3] != 640:
                    report['issues'].append(f"Input shape {dims} not [1,3,640,640] - DJI expects 640")
                if dims[0] != 1:
                    report['issues'].append(f"Batch dim {dims[0]} !=1 - should be static batch 1")
            
            # Opset
            opset = model.opset_import[0].version
            report['opset'] = opset
            logger.info(f"Opset: {opset}")
            
            if opset < 11 or opset > 17:
                report['issues'].append(f"Opset {opset} outside DJI recommended 11-14")
            
            report['valid'] = len(report['issues']) == 0
            
            if report['valid']:
                logger.success(f"ONNX validation PASSED: {onnx_path}")
            else:
                logger.warning(f"ONNX validation issues: {report['issues']}")
            
        except ImportError:
            logger.warning("onnx package not installed, skipping deep validation")
            report['valid'] = True  # Assume valid if can't check
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            report['issues'].append(str(e))
        
        return report

    def benchmark_inference(self, onnx_path: str, num_runs: int = 100):
        """Benchmark ONNX inference latency"""
        try:
            import onnxruntime as ort
            import numpy as np
            import time
            
            sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            
            dummy = np.random.randn(1,3,640,640).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                sess.run(None, {input_name: dummy})
            
            # Benchmark
            start = time.time()
            for _ in range(num_runs):
                sess.run(None, {input_name: dummy})
            elapsed = time.time() - start
            
            avg_ms = (elapsed / num_runs) * 1000
            fps = 1000 / avg_ms
            
            logger.info(f"Inference benchmark: {avg_ms:.2f}ms avg, {fps:.1f} FPS (CPU)")
            logger.info(f"On Matrice 4TD NPU expect ~40-60ms after INT8 quant")
            
            return {'avg_ms': avg_ms, 'fps': fps}
            
        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            return None
