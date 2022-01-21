from __future__ import print_function

import os
import ctypes
import argparse

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np

import common
from timer import DTimer

try:
    ctypes.cdll.LoadLibrary('libyolo_layer.so')
except OSError as e:
    raise SystemExit('ERROR: failed to load libyolo_layer.so.') from e

logger = trt.Logger()
runtime = trt.Runtime(logger)

batch_size = 1
model_path = f'...'

with open('%s.trt' % model_path, 'rb') as f:
    serialized_engine = f.read()
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()

inputs, outputs, bindings, stream = common.allocate_buffers(engine)

iters = 10

time_sum = 0
for iter in range(iters+1):
    _in = np.random.uniform(size=engine.get_binding_shape(engine[0])).astype(np.float32)
    cuda.memcpy_htod(inputs[0].device, _in)

    with DTimer() as time:
        trt_outputs = common.do_inference_v2(context, bindings=bindings,
                                             inputs=inputs, outputs=outputs,
                                             stream=stream)
    print(f"iter {iter}: {time.elapsed}")
    if iter > 0:
        time_sum += time.elapsed.total_seconds()
time_avg = time_sum / (iters)
print(f"avg: {time_avg} (sec/batch)\n")
print(f"FPS: {batch_size / time_avg}")
