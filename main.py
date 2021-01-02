import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from src.infer import Blur_Background

if __name__ == "__main__":
    obj = Blur_Background
    args = obj.arg_parser()
    interpreter = tflite.Interpreter(model_path=args.model)
    input_det, output_det = obj.allocate_tensor(interpreter)

    vid = cv2.VideoCapture(args.cam_id)

    while(vid.isOpened()):
        ret, frame = vid.read()
        cpy = np.copy(frame)
        arr = obj.infer(interpreter, frame, input_det, output_det)
        frame = obj.blur(arr, cpy, frame)

        cv2.imwrite('output/out.jpg', frame)


    vid.release()

