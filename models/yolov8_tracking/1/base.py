import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadPilAndNumpy
from ultralytics.yolo.utils.checks import check_imgsz, check_requirements
from ultralytics.yolo.utils.ops import (
    non_max_suppression,
    process_mask,
    process_mask_native,
    scale_boxes,
)
from ultralytics.yolo.utils.torch_utils import select_device
from trackers.multi_tracker_zoo import create_tracker


ROOT = Path(os.environ['ROOT_FOLDER'])
WEIGHTS = ROOT / "weights"



class Yolov8Tracking:
    @torch.no_grad()
    def __init__(
        self,
        yolo_weights=WEIGHTS / "yolov8n-seg.pt",  # model.pt path(s),
        reid_weights=WEIGHTS / "osnet_x0_25_msmt17.pt",  # model.pt path,
        tracking_method="botsort",
        imgsz=(640, 640),  # inference size (height, width)
        augment=False,
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        half=False,  # use FP16 half-precision inference
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        dnn=False,  # use OpenCV DNN for ONNX inference
        classes=None,  # filter by class: --class 0, or --class 0 2 3
    ) -> None:
        self.bs = 1
        self.device = select_device(device)
        self.is_seg = "-seg" in str(yolo_weights)
        self.model = AutoBackend(yolo_weights, device=self.device, dnn=dnn, fp16=half)
        self.stride, self.names, self.pt = (
            self.model.stride,
            self.model.names,
            self.model.pt,
        )
        self.imgsz = check_imgsz(imgsz, stride=self.stride)  # check image size
        self.model.warmup(
            imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz)
        )  # warmup
        self.augment = augment

        # Create as many strong sort instances as there are video sources
        bs = 1
        self.tracker_list = []
        tracking_config = (
            ROOT
            / "trackers"
            / tracking_method
            / "configs"
            / (tracking_method + ".yaml")
        )
        for i in range(bs):
            tracker = create_tracker(
                tracking_method, tracking_config, reid_weights, self.device, half
            )
            self.tracker_list.append(
                tracker,
            )
            if hasattr(self.tracker_list[i], "model"):
                if hasattr(self.tracker_list[i].model, "warmup"):
                    self.tracker_list[i].model.warmup()

        self.half = half
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        agnostic_nms=False,
        retina_masks=False,
    ):
        dataset = LoadPilAndNumpy(
            im0=image,
            imgsz=self.imgsz,
            auto=self.pt,
            stride=self.stride,
            transforms=getattr(self.model.model, "transforms", None),
        )
        outputs = [None] * self.bs
        seen = 0
        curr_frames, prev_frames = [None] * self.bs, [None] * self.bs
        for _, batch in enumerate(dataset):
            _, im, im0s, _, _ = batch
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            preds = self.model(im, augment=self.augment, visualize=False)

            # Apply NMS

            if self.is_seg:
                masks = []
                p = non_max_suppression(
                    preds[0],
                    self.conf_thres,
                    self.iou_thres,
                    self.classes,
                    agnostic_nms,
                    max_det=self.max_det,
                    nm=32,
                )
                proto = preds[1][-1]
            else:
                p = non_max_suppression(
                    preds,
                    self.conf_thres,
                    self.iou_thres,
                    self.classes,
                    agnostic_nms,
                    max_det=self.max_det,
                )

            # Process detections
            for i, det in enumerate(p):  # detections per image
                seen += 1

                if isinstance(im0s, list):
                    im0 = im0s[i].copy()

                else:
                    im0 = im0s.copy()
                curr_frames[i] = im0
                if hasattr(self.tracker_list[i], "tracker") and hasattr(
                    self.tracker_list[i].tracker, "camera_update"
                ):
                    if (
                        prev_frames[i] is not None and curr_frames[i] is not None
                    ):  # camera motion compensation
                        self.tracker_list[i].tracker.camera_update(
                            prev_frames[i], curr_frames[i]
                        )

                if det is not None and len(det):
                    if self.is_seg:
                        shape = im0.shape
                        # scale bbox first the crop masks
                        if retina_masks:
                            det[:, :4] = scale_boxes(
                                im.shape[2:], det[:, :4], shape
                            ).round()  # rescale boxes to im0 size
                            masks.append(
                                process_mask_native(
                                    proto[i], det[:, 6:], det[:, :4], im0.shape[:2]
                                )
                            )  # HWC
                        else:
                            masks.append(
                                process_mask(
                                    proto[i],
                                    det[:, 6:],
                                    det[:, :4],
                                    im.shape[2:],
                                    upsample=True,
                                )
                            )  # HWC
                            det[:, :4] = scale_boxes(
                                im.shape[2:], det[:, :4], shape
                            ).round()  # rescale boxes to im0 size
                    else:
                        det[:, :4] = scale_boxes(
                            im.shape[2:], det[:, :4], im0.shape
                        ).round()  # rescale boxes to im0 size

                    # pass detections to strongsort
                    outputs[i] = self.tracker_list[i].update(det.cpu(), im0)

                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        for _, (output) in enumerate(outputs[i]):
                            bbox = output[0:4]
                            id = output[4]
                            cls = output[5]
                            conf = output[6]

                prev_frames[i] = curr_frames[i]

        return outputs


if __name__ == "__main__":
    check_requirements(
        requirements=ROOT / "requirements.txt", exclude=("tensorboard", "thop")
    )
    img = cv2.imread(
        "/home/michou/Project/yolov8_tracking/14crowd-1-1-069d-videoSixteenByNine3000.jpg"
    )
    obj = Yolov8Tracking()
    for i in range(10):
        result = obj.predict(image=img)
        print(result)
        result = np.asarray(result)
        print(result)
        print(result[..., 0:3])
        print(result[..., 4])
        print(result[..., 5])
        print(result[..., 6])
