import os


MODEL_PARAM = {
    # IMG_SIZE FOR INFERENCE
    "IMG_SIZE": int(os.environ.get("IMG_SIZE", "640")),
    # PATH_TO_MODEL
    "MODEL": os.environ.get("MODEL", "yolov8n.pt"),
    "REID_MODEL" : os.environ.get("REID_MODEL", "osnet_x0_25_msmt17.pt"),
    "TRACK_METHOD" : os.environ.get('TRACK_METHOD', "botsort")
    # GPU ID OR CPU
    "DEVICE_ID": os.environ.get("DEVICE_ID", "cpu"),
    # HALF for F16 precision
    "HALF": bool(os.environ.get("HALF", "FALSE").lower() == "true"),
}

INFERENCE_PARAM = {
    # Prediction Threshold
    "INFERENCE_CONF": float(os.environ.get("INFERENCE_CONF", "0.5")),
    # NMS Threshold
    "NMS_IOU": float(os.environ.get("NMS_IOU", "0.45")),
    "MAX_DET": int(os.environ.get("MAX_DET", "100"))
    # Agnostic NMS Parameter
    "AGNOSTIC_NMS": bool(os.environ.get("AGNOSTIC_NMS", "FALSE").lower() == "true"),
}