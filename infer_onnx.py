import onnxruntime
import cv2
import numpy as np
import torch
from utils import overlay, segment_everything
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from PIL import Image
from random import randint
import time

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

def pre_processing(img_origin, imgsz=1024):
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
    img = cv2.imread(image_path)
    size_img = 1024
    inp = pre_processing(img)
    #inp = process(image_path, (1024, 1024))
    print('Input: ', inp.shape)
    
    model = onnxruntime.InferenceSession('./models/fast_sam_1024.onnx')
    #inp = np.ones([1, 3, size_img, size_img]).astype(np.float32)
    start = time.time()
    ort_inputs = {model.get_inputs()[0].name: inp}
    preds = model.run(None, ort_inputs)
    end = time.time()
    print(end-start, 's')
    print(preds[0])
    
    print([x.shape for x in preds])
    data_0 = torch.from_numpy(preds[0])
    data_1 = [[torch.from_numpy(preds[1]), torch.from_numpy(preds[2]), torch.from_numpy(preds[3])], torch.from_numpy(preds[4]), torch.from_numpy(preds[5])]
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
    cv2.imwrite("./outputs/onnx_infer.png", image_with_masks)