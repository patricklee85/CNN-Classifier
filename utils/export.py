
import copy
import torch

def export_onnx(model_pt, input_shape, output_folder):
    # save best.onnx
    best_model_onnx = copy.deepcopy(model_pt).to('cpu')
    save_path = f'{output_folder}\\best.onnx'
    torch.onnx.export(
        best_model_onnx,                                # model being run
        torch.randn(input_shape),                       # model input (or a tuple for multiple inputs)
        save_path,                                      # where to save the model (can be a file or file-like object)
        export_params=True,                             # store the trained parameter weights inside the model file
        opset_version=11,                               # the ONNX version to export the model to
        do_constant_folding=True,                       # whether to execute constant folding for optimization
        input_names = ['input'],                        # the model's input names
        output_names = ['output'],                      # the model's output names
        dynamic_axes={
            'input' : {0 : 'batch_size'}, 
            'output' : {0 : 'batch_size'}})             # variable length axes

