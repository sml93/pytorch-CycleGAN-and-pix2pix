#!usr/bin/env python
# USAGE: python webcam_cyclegan.py --name MODEL_NAME --model test --preprocess none --no_dropout

import os
import cv2
import torch
import imutils
import numpy as np
import pyrealsense2 as rs

from util import util
from models import create_model
from data import create_dataset
from options.test_options import TestOptions


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # Start video
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start visual sensor
    pipeline.start(config)

    data = {"A": None, "A_paths": None}

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        frame = imutils.resize(color_image, width=480)
        h, w, c = frame.shape

        frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.array([frame])
        frame = frame.transpose([0,3,1,2])
        data['A'] = torch.FloatTensor(frame)

        model.set_input(data)
        model.test()

        result_image = model.get_current_visuals()['fake']
        result_image = util.tensor2im(result_image)
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_BGR2RGB)
        result_image = cv2.resize(result_image, (512, 512))
        # result_image = cv2.putText(result_image, str(opt.name)[6:-11], org, font,
        #                            fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Result", result_image)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
