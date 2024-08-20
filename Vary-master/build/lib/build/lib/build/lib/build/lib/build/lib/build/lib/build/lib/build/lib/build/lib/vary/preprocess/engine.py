import cv2
import dewarping2
import numpy as np


IMG_SIZES = [(1152,1536),(1792,2400),(1536,2048),]




class Engine(object):

    def __init__(self, triton_client):
        self.triton_client = triton_client

    def predict(self, img, quality_level=0, mode=2, resize_to_minsize=False, shm_mode=1):
        h, w = img.shape[:2]

        if w > h:
            img = np.rot90(img)
        inference_size = IMG_SIZES[quality_level]
        LEAST_IMG_SIZE = inference_size[0]

        left = 0
        right = 0
        if img.shape[0] / img.shape[1] > 2 and mode < 3:
            img = cv2.resize(img, (inference_size[1]*img.shape[1]//img.shape[0], inference_size[1]))
            left = (inference_size[0] - img.shape[1]) // 2
            right = inference_size[0] - left - img.shape[1]
            img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            img = cv2.resize(img, inference_size)

        if mode == 0:
            output_byte_size = img.shape[0] * img.shape[1] * 3
            output_image = self.triton_client.infer(img, 'hw_model', output_byte_size, shm_mode=shm_mode)
        elif mode == 1:
            output_byte_size = img.shape[0] * img.shape[1]
            output_image = self.triton_client.infer(img, 'pp_model', output_byte_size, shm_mode=shm_mode)
        elif mode == 2:
            output_byte_size = img.shape[0] * img.shape[1]
            output_image = self.triton_client.infer2(img, 'hw_model', output_byte_size * 3, 'pp_model', output_byte_size,
                                                     shm_mode=shm_mode)
        elif mode == 3:
            output_byte_size = img.shape[0] * img.shape[1] * 3
            output_image = self.triton_client.infer(img, 'hw_model', output_byte_size, shm_mode=shm_mode)
            output_image = dewarping2.balanceWhite(output_image)
            output_image = self.triton_client.infer(output_image, 'color_model', output_byte_size, shm_mode=shm_mode)
        elif mode == 4:
            output_byte_size = img.shape[0] * img.shape[1] * 3
            img = dewarping2.balanceWhite(img)
            output_image = self.triton_client.infer(img, 'color_model', output_byte_size, shm_mode=shm_mode)
        elif mode == 6:
            output_byte_size = img.shape[0] * img.shape[1]
            output_image = self.triton_client.infer(img, 'ar_hw_pp_model', output_byte_size, input_name='input',
                                                    output_name='output', shm_mode=shm_mode)
        elif mode == 7:
            output_byte_size = img.shape[0] * img.shape[1] * 3
            output_image = self.triton_client.infer(img, 'ar_hw_color_model', output_byte_size, input_name='input',
                                                    output_name='output', shm_mode=shm_mode)

        if left != 0 or right != 0:
            output_image = output_image[:, left:(inference_size[0]-right)]

        if mode != 0:
            if not output_image.flags.writeable:
                output_image = output_image.copy()
            output_image[output_image >= 250] = 255

        if w > h:
            output_image = np.rot90(output_image, 3)

        # incase of small cropped pic, to preserve quality
        if resize_to_minsize:
            if h < w and w < LEAST_IMG_SIZE:
                h = int(h * LEAST_IMG_SIZE / w)
                w = LEAST_IMG_SIZE
            if h >= w and h < LEAST_IMG_SIZE:
                w = int(w * LEAST_IMG_SIZE / h)
                h = LEAST_IMG_SIZE
        return cv2.resize(output_image, (w, h))


if __name__ == "__main__":
    import glob
    import os
    import io

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    engine = Engine("./models/hw_model","./models/pp_model")

    test_files = glob.glob("./local/testcases/normal.jpg")
    #test_files = ["/mnt/data1/datasets/hwremove/httests/*.jpg"]
    for f in test_files:
        img = cv2.imread(f)

        output_image = engine.predict(img,1,2)
        cv2.imwrite("/mnt/data1/datasets/hwremove/output/output.png",output_image)
        continue
        ''' 
        _, jpg_buf = cv2.imencode(".jpg", output_image)
        im_bytes = io.BytesIO()
        np.savez(im_bytes, img=jpg_buf)
        im_bytes.seek(0)
        '''
        output_image = imutils.resize(output_image, height=1000)
        cv2.imshow("img", output_image)
        cv2.waitKey()
