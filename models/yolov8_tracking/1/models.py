import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from config import MODEL_PARAM, INFERENCE_PARAM
from base import Yolov8Tracking, WEIGHTS


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        """

        # Load YOLOv5s
        self.model = Yolov8Tracking(
            yolo_weights= WEIGHTS / MODEL_PARAM['MODEL'], 
            reid_weights= WEIGHTS / MODEL_PARAM['REID_MODEL'],
            tracking_method=MODEL_PARAM['TRACK_METHOD'],
            imgsz=(MODEL_PARAM['IMG_SIZE'], MODEL_PARAM['IMG_SIZE']),
            augment=False,conf_thres=INFERENCE_PARAM['INFERENCE_CONF'], iou_thres=INFERENCE_PARAM['NMS_IOU'], 
            max_det=INFERENCE_PARAM['MAX_DET'], half=INFERENCE_PARAM['HALF'], device=MODEL_PARAM['DEVICE_ID'],
            dnn=False, 
            classes=None, 

        )

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        for request in requests:
            in_ = pb_utils.get_input_tensor_by_name(request, "INPUT")
            in_ = in_.as_numpy()[..., ::-1]

            in_ = [
                el for el in in_
            ]  # TODO: Need to investigate about the batch management

            return_output = self.model.predict(in_[0])

            return_output_array = np.asarray(
                [t.detach().numpy() for t in return_output]
            )
            # print(result[..., 0:3])
            # print(result[..., 4])
            # print(result[..., 5])
            # print(result[..., 6])
            # bbox = output[0:4]
            # id = output[4]
            # cls = output[5]
            # conf = output[6]

            # return_output = pb_utils.Tensor("OUTPUT", return_output)
            tensor_bbx = pb_utils.Tensor("BBX", return_output_array[:, :, :4])
            tensor_ids = pb_utils.Tensor("IDS", return_output_array[:, :, 4])
            tensor_class = pb_utils.Tensor("CLASS", return_output_array[:, :, 5])
            tensor_score = pb_utils.Tensor("SCORE", return_output_array[:, :, 6])

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))

            # inference_response = pb_utils.InferenceResponse(output_tensors=[return_output])
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tensor_bbx, tensor_score, tensor_class, tensor_ids]
            )
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")