import numpy as np
import cv2
import argparse 
import tflite_runtime.interpreter as tflite

class Blur_Background: 

    def allocate_tensor(interpreter):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.resize_tensor_input(output_details[0]["index"],[1, 240, 320, 1])
        interpreter.resize_tensor_input(input_details[0]["index"],[1, 240, 320, 3])

        interpreter.allocate_tensors()

        return input_details, output_details

    def infer(interpreter, frame, input_details, output_details):
        new_img = cv2.resize(frame, (320,240))
        input_data = np.array(new_img, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], [input_data])

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.reshape(output_data, (240,320))
        arr = np.array(output_data*255.0, dtype="uint8")

        return arr

    def blur(arr, cpy, frame):
        cpy = cv2.blur(cpy, (25, 25))
        ret,th1 = cv2.threshold(arr,70,255,cv2.THRESH_BINARY)

        map = th1
        map = cv2.resize(map, (frame.shape[1], frame.shape[0]))
        
        frame[:,:,0] = frame[:,:,0] & map
        frame[:,:,1] = frame[:,:,1] & map
        frame[:,:,2] = frame[:,:,2] & map

        imap = ~map

        cpy[:,:,0] = cpy[:,:,0] & imap
        cpy[:,:,1] = cpy[:,:,1] & imap
        cpy[:,:,2] = cpy[:,:,2] & imap

        frame = frame | cpy

        return frame


    def arg_parser():
        arg_parser = argparse.ArgumentParser(description='Background blur')

        arg_parser.add_argument('--model', metavar='model_name', type=str, help='Input file name', required=True)
        arg_parser.add_argument('--cam_id', metavar='Device index', type=int, help='Pass the device index to capture live stream', default=0)

        args = arg_parser.parse_args()

        return args

