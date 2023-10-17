import onnx
from onnxsim import simplify

model_path = "./models/fast_sam_1024.onnx"
save_path = "./models/fast_sam_1024_simply.onnx"

print('模型静态图开始简化')
model_onnx = onnx.load_model(model_path)
model_smi, check = simplify(model_onnx)
onnx.save(model_smi, save_path)
print('模型静态图简化完成')
