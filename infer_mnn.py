import onnxruntime
import cv2
import numpy as np
import torch
from utils import overlay, segment_everything
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from PIL import Image
from random import randint
import MNN
import MNN.expr as F
import time
import psutil

retina_masks = True
conf = 0.25
iou = 0.7
agnostic_nms = False

def postprocess(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=False):
    """TODO: filter by classes."""
    
    p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=100,
                                nc=1)



    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        # path = self.batch[0]
        img_path = "ok"
        if not len(pred):  # save empty boxes
            results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names="1213", boxes=pred[:, :6], masks=masks))
    return results

def pre_processing(img_origin, imgsz=1024
                   ):
    h, w = img_origin.shape[:2]
    if h>w:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nh-nw)/2) 
        inp[: nh, a:a+nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nw-nh)/2) 

        inp[a: a+nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype = np.float32) / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))

# 图像预处理
def process(image_path, size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_resize = cv2.resize(image, size).astype(float)
    image_resize /= 255
    input_data = np.array(image_resize)
    input_data = input_data.transpose((2, 0, 1))  # HWC --> CHW
    input_data = np.expand_dims(input_data, 0)
    return input_data

if __name__ == '__main__':
    image_path = './images/cat.jpg'
    mnnmodel = './models/convert_int8.mnn' # './models/mnn_fp32.mnn' 
    print('model: ', mnnmodel)
    img = cv2.imread(image_path)
    size_img = 1024
    inp = pre_processing(img, size_img)
    #inp = process(image_path, (size_img, size_img))
    print('Input: ', inp.shape)
    
    interpreter = MNN.Interpreter(mnnmodel)
    mnn_session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(mnn_session)
    input_data1 = MNN.Tensor((1, 3, size_img, size_img), MNN.Halide_Type_Float, inp, MNN.Tensor_DimensionType_Tensorflow) # MNN.Tensor_DimensionType_Caffe
    input_data1 = F.const(input_data1.getHost(), input_data1.getShape(), F.NCHW) # NHWC, NCHW, NC4HW4
    tmp_input = MNN.Tensor(input_data1)

    interpreter.resizeTensor(input_tensor, (1, 3, size_img, size_img))
    interpreter.resizeSession(mnn_session)
    
    #input_tensor = inp
    input_tensor.copyFrom(tmp_input)
    
    #mnn infer
    start=time.time()
    interpreter.runSession(mnn_session)
    print('mnn infer total time is %.4f s'%(time.time()-start))
    print(psutil.cpu_percent(interval=1))
    output_tensor = interpreter.getSessionOutputAll(mnn_session)

    # 从输出Tensor拷贝出数据 
    output_data0 = MNN.Tensor(output_tensor['output0'].getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    output_tensor['output0'].copyToHostTensor(output_data0)
    data0 = output_data0.getNumpyData()
    print(data0)

    output_data1 = MNN.Tensor(output_tensor['output1'].getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    output_tensor['output1'].copyToHostTensor(output_data1)
    data1 = output_data1.getNumpyData()
    
    output_data2 = MNN.Tensor(output_tensor['onnx::Reshape_1252'].getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    output_tensor['onnx::Reshape_1252'].copyToHostTensor(output_data2)
    data2 = output_data2.getNumpyData()

    output_data3 = MNN.Tensor(output_tensor['onnx::Reshape_1271'].getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    output_tensor['onnx::Reshape_1271'].copyToHostTensor(output_data3)
    data3 = output_data3.getNumpyData()

    output_data4 = MNN.Tensor(output_tensor['onnx::Concat_1213'].getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    output_tensor['onnx::Concat_1213'].copyToHostTensor(output_data4)
    data4 = output_data4.getNumpyData()

    output_data5 = MNN.Tensor(output_tensor['1167'].getShape(), MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
    output_tensor['1167'].copyToHostTensor(output_data5)
    data5 = output_data5.getNumpyData()
    
    print('output data print over!')

    print(data0.shape, data1.shape, data2.shape, data3.shape, data4.shape, data5.shape)
    data_0 = torch.from_numpy(data0)
    data_1 = [[torch.from_numpy(data1), torch.from_numpy(data2), torch.from_numpy(data3)], torch.from_numpy(data4), torch.from_numpy(data5)]
    preds = [data_0, data_1]
    result = postprocess(preds, inp, img, retina_masks, conf, iou)
    masks = result[0].masks.data
    print("len of mask: ", len(masks))
    image_with_masks = np.copy(img)
    for i, mask_i in enumerate(masks):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        image_with_masks = overlay(image_with_masks, mask_i, color=rand_color, alpha=1)
    cv2.imwrite("./outputs/mnn_infer.png", image_with_masks)
    print('success!')