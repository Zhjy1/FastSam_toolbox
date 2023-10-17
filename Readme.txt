代码文件说明：
pt2onnx.sh、ptonnx.py：模型转换代码（由pytorch模型转为onnx模型）
onnx2trt.sh：onnx模型转为trt模型的代码

onnx_simplify.py：onnx模型简化代码
onnx_quant.py：onnx模型量化代码

infer_mnn.py：mnn模型推理代码
infer_onnx.py：onnx模型推理代码
inference_trt.py：trt模型推理代码

代码使用说明：
1、FastSAM模型由pt转为onnx格式
python pt2onnx.py --weights "./models/FastSAM_X.pt" --output "./models/fastsam_1024.onnx" --max_size 1024

2、编译mnn（在linux上完成）
git clone mnn-master
cd mnn-master
mkdir build
cd build
cmake -DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_CONVERTER=true ..
make -j4

3、onnx格式转为mnn格式（接着2的编译结果）
./build/MNNConvert -f ONNX --modelFile ./models/fastsam_1024.onnx --MNNModel ./models/fastsam_mnn_fp32.mnn --bizCode biz --testdir ./data2
检验由onnx转换为mnn时是否成功：
python ../tools/script/testMNNFromOnnx.py ../models/fastsam_1024.onnx

4、mnn格式下进行模型的单输入离线量化，fp32->int8
./build/quantized.out ./models/fastsam_mnn_fp32.mnn ./models/fastsam_mnn_int8.mnn ./imageInputConfig.json

python的模型单输入离线量化：
mnnquant fastsam_mnn_fp32.mnn fastsam_mnn_int8.mnn imageInputConfig.json
python的模型多输入离线量化：
python mnn_offline_quant.py --mnn_model fastsam_mnn_fp32.mnn --quant_model fastsam_mnn_int8.mnn --batch_size 1
(python的并没有跑通，可作为备选）

转换过程中直接进行量化：
fp32->fp16：
./build/MNNConvert -f ONNX --fp16 --modelFile ./models/fastsam_1024.onnx --MNNModel ./models/fastsam_mnn_fp16.mnn --bizCode biz --testdir ./onnx
fp32->int8：
./build/MNNConvert -f ONNX --weightQuantBits 8 --modelFile ./models/fastsam_1024.onnx --MNNModel ./models/fastsam_mnn_int8.mnn --bizCode biz --testdir ./onnx

5、mnn模型推理（FastSam_model_process-main中）
python infer_mnn.py

6、测试平均耗时（在2编译结果的基础上，对models文件下的所有模型进行统计）
./build/benchmark.out ./models 10 0

P.S：
获取mnn模型的详细信息：
./build/GetMNNInfo ./models/2nd_1024.mnn 
FastSAM模型的输入输出信息：
	images：(1, 3, 1024, 1024, )
	output0: (1, 37, 21504, )
	output1: (1, 105, 128, 128, )
	onnx::Reshape_1252: (1, 105, 64, 64, )
	onnx::Reshape_1271: (1, 105, 32, 32, )
	onnx::Concat_1213: (1, 32, 21504, )
	1167: (1, 32, 256, 256, )
