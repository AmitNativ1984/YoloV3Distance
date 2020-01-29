import torch
import onnx
from onnx import optimizer
import onnxruntime
import numpy as np
import logging

def pytorch2onnx(model, input_size, onnx_model_path):
    """ convert pytorch model to onnx. save onnx to export_path"""

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    x = torch.randn(input_size, requires_grad=True)

    model.eval()
    torch_out1, torch_out2 = model(x)   # pytorch output. will be used for veryfiction

    # export the model
    logging.info("converting pytorch model to onnx...")
    torch.onnx.export(model,                # input pytorch model
                      x,                    # model input
                      onnx_model_path,  # path where onnx model is saved
                      verbose=False,
                      opset_version=10)     # the ONNX version to export the model to

    logging.info("convertion to onnx completed successfully")

    # verifying conversion was successfull
    onnx_model = onnx.load(onnx_model_path)
    logging.info("onnx model loaded successfully")
    # onnx.checker.check_model(onnx_model)
    logging.info("onnx model saved successfully to: {}".format(onnx_model_path))

    # comparing result to pytorch results:
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out1), ort_outs[0], rtol=1e-03, atol=1e-05)

    logging.info("Exported model has been tested with ONNXRuntime, and the result looks good!")