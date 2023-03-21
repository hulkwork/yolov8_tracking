import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from config import INFERENCE_PARAM, MODEL_PARAM, yolov5_path


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
        self.model = torch.hub.load(
            yolov5_path, "custom", path=MODEL_PARAM["MODEL_PATH"], source="local"
        )

        # Configuration of the model
        self.model.conf = INFERENCE_PARAM["INFERENCE_CONF"]
        self.model.iou = INFERENCE_PARAM["NMS_IOU"]
        self.model.agnostic = INFERENCE_PARAM["AGNOSTIC_NMS"]

        # Load on CPU if needed otherwise load on the right device
        self.model.cpu() if MODEL_PARAM["DEVICE_ID"] == "cpu" else self.model.to(
            MODEL_PARAM["DEVICE_ID"]
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

            return_output = self.model(in_, size=MODEL_PARAM["IMG_SIZE"]).xyxy

            return_output_array = np.asarray(
                [t.detach().numpy() for t in return_output]
            )

            # return_output = pb_utils.Tensor("OUTPUT", return_output)
            tensor_bbx = pb_utils.Tensor("BBX", return_output_array[:, :, :4])
            tensor_score = pb_utils.Tensor("SCORE", return_output_array[:, :, 4])
            tensor_class = pb_utils.Tensor("CLASS", return_output_array[:, :, 5])

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))

            # inference_response = pb_utils.InferenceResponse(output_tensors=[return_output])
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tensor_bbx, tensor_score, tensor_class]
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