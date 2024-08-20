import numpy as np
import cv2
import time
import dewarping2


class DewarpEngine(object):

    def __init__(self, triton_client):
        self.triton_client = triton_client
        self.w = 576
        self.h = 768
        self.w2 = 576
        self.h2 = 896

    def predict(self, img, img2=None):
        h, w = img.shape[:2]
        if h < w * 2 and w < h * 4:
            if w > h:
                if w / h > self.w2 / self.h2:
                    new_h = self.w2 * h // w
                    input_image = cv2.resize(img, (self.w2, new_h))
                    pad = self.h2 - new_h
                    input_image = cv2.copyMakeBorder(input_image, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT,
                                                     value=[255, 255, 255])
                else:
                    new_w = self.h2 * w // h
                    pad = self.w2 - new_w
                    input_image = cv2.resize(img, (new_w, self.h2))
                    input_image = cv2.copyMakeBorder(input_image, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT,
                                                     value=[255, 255, 255])
            else:
                input_image = cv2.resize(img, (self.w2, self.h2))
            if len(input_image.shape) == 2 or (len(input_image.shape) == 3 and input_image.shape[2] == 1):
                input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
            output_byte_size = input_image.shape[0] * input_image.shape[1] * 3
            output_image = self.triton_client.infer(input_image, 'dewarp2_model', output_byte_size,
                                                    'INPUT__0', np.uint8,
                                                    'OUTPUT__0', np.uint8)
            if w > h:
                if w / h > self.w2 / self.h2:
                    output_image = output_image[pad // 2:self.h2 - pad + pad // 2, :]
                else:
                    output_image = output_image[:, pad // 2:self.w2 - pad + pad // 2]
            output_image = dewarping2.dewarp(img if img2 is None else img2, output_image[:, :, ::-1], 2, 0x3f)
            if output_image is not None:
                return output_image

        top = bottom = 64
        input_image = cv2.resize(img, (self.w, self.h-top-bottom))
        input_image = cv2.copyMakeBorder(input_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        if len(input_image.shape) == 2 or (len(input_image.shape) == 3 and input_image.shape[2] == 1):
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        for j in range(32, self.h, 32):
            cv2.line(input_image, (0, j), (self.w, j), (255, 0, 0), 3)

        output_byte_size = input_image.shape[0] * input_image.shape[1] * 3
        output_image = self.triton_client.infer(input_image, 'dewarp_model', output_byte_size,
                                                'INPUT__0', np.uint8,
                                                'OUTPUT__0', np.uint8)
        output_image = output_image[:, :, ::-1]
        pad = img.shape[0]*top//(self.h-top-bottom)
        img = cv2.copyMakeBorder(img if img2 is None else img2, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        output_image = dewarping2.dewarp(img, output_image, 0, 0)
        output_image = output_image[pad:img.shape[0]-pad, :]
        return output_image

    def balance(self, img):
        return dewarping2.balanceWhite(img)


if __name__ == '__main__':
    img = cv2.imread('1.png')
    pad = img.shape[0] * 64 // (768 - 128)
    img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    output_image = cv2.imread('3.png')
    output_image = dewarping2.dewarp(img, output_image)
    output_image = output_image[pad:img.shape[0] - pad, :]
    cv2.imwrite('5.png', output_image)
