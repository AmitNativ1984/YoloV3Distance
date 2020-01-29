import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(args):
    model_path = args.model_path
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = 1 << 20
        builder.max_batch_size = 1
        with open(model_path, "rb") as f:
            parser.parse(f.read())
        engine = builder.build_cuda_engine(network)
        return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream

def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/home/amit/Dev/Kzir/weights/export.onnx', help='cfg file path')
    parser.add_argument('--input-size', type=int, default='416', help='input img size')

    args = parser.parse_args()

    inputs = np.random.random((1, 3, args.input_size, args.input_size)).astype(np.float32)
    engine = build_engine(args)
    context = engine.create_execution_context()
    for _ in range(10):
        t1 = time.time()
        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
        print(res)