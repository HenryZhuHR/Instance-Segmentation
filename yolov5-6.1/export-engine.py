import os
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def main():
    onnx_model_file='weights/yolov5s.onnx'
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    workspace =4
    config.max_workspace_size = workspace *1 << 30 # Your workspace size
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse_from_file(str(onnx_model_file)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_model_file}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    engine_file_path=os.path.splitext(onnx_model_file)[0]+'.engine'

    
    if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(engine_file_path, 'wb') as t:
        t.write(engine.serialize())
    print('save to:',engine_file_path)

if __name__=='__main__':
    main()  