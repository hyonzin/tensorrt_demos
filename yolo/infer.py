from __future__ import print_function

import os
import argparse

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np

import common
import plugins
from timer import DTimer

logger = trt.Logger()
runtime = trt.Runtime(logger)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fp', type=int, required=True)
parser.add_argument('-b', '--bs', type=int, required=True)
args = parser.parse_args()

print(f'Batch size {args.bs}, FP{args.fp}')

args_model = f'/home/hjjung/work/models/yolov4-modanet.darknet/yolov4-modanet-416-maxbs{args.bs}-fp{args.fp}'

with open('%s.trt' % args_model, 'rb') as f:
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
print(f"avg: {time_avg} (sec)\n")
print(f"FPS: {args.bs / time_avg}")
