import torch.onnx 
import torch 
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

#Function to Convert to ONNX 
def Convert_ONNX(model): 
    dummy_input = torch.randn((1,3,640,640), requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == "__main__": 
    model = build_model('./Partitioned_COIM_R50_6x+2x.yaml')
    model.load_state_dict(torch.load('./Partitioned_COIM_R50_6x+2x.pth',map_location ='cpu'))
    Convert_ONNX(model) 