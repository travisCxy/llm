import os, uuid
import random
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.shared_memory as shm
# import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient.utils import np_to_triton_dtype

class TritonClient(object):

    def __init__(self):
        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if devices is None:
            self.devices = [0]
        else:
            devices = devices.split(',')
            if len(devices) > 0:
                self.devices = [i for i in range(len(devices))]
            else:
                self.devices = [0]

    def create_shm(self, client, name, size, use_cuda):
        if use_cuda:
            random.shuffle(self.devices)
            handle = None
            for i in range(len(self.devices)):
                try:
                    device_id = self.devices[i]
                    handle = cudashm.create_shared_memory_region(name, size, device_id)
                    client.register_cuda_shared_memory(name, cudashm.get_raw_handle(handle), device_id, size)
                    break
                except:
                    pass
        else:
            key = name.replace('-data', '-key')
            handle = shm.create_shared_memory_region(name, key, size)
            client.register_system_shared_memory(name, key, size)
            client.get_system_shared_memory_status(name, as_json=True)
        assert handle is not None
        return handle

    def destroy_shm(self, client, name, handle, use_cuda):
        if use_cuda:
            client.unregister_cuda_shared_memory(name=name)
            cudashm.destroy_shared_memory_region(handle)
        else:
            client.unregister_system_shared_memory(name=name)
            shm.destroy_shared_memory_region(handle)

    def infer(self, img, model_name, output_byte_size,
              input_name='input_2', idtype=np.uint8,
              output_name='tf.cast_1', odtype=np.uint8,
              shm_mode=1):
        input_data = np.expand_dims(img, axis=0)
        if input_data.dtype != idtype:
            input_data = input_data.astype(idtype)
        dtype = np_to_triton_dtype(idtype)

        triton_client = grpcclient.InferenceServerClient(url='localhost:8001')
        if shm_mode > 0:
            use_cuda = shm_mode > 1
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = input_data.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, use_cuda)

            output_shm_name = "output-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            shm_op_handle = self.create_shm(triton_client, output_shm_name, output_byte_size, use_cuda)

            if use_cuda:
                cudashm.set_shared_memory_region(shm_ip_handle, [input_data])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [input_data])
            inputs = [grpcclient.InferInput(input_name, input_data.shape, dtype)]
            inputs[-1].set_shared_memory(input_shm_name, input_byte_size)

            outputs = [grpcclient.InferRequestedOutput(output_name)]
            outputs[-1].set_shared_memory(output_shm_name, output_byte_size)

            results = triton_client.infer(model_name, inputs, outputs=outputs)

            output = results.get_output(output_name)
            if use_cuda:
                output_data = cudashm.get_contents_as_numpy(shm_op_handle, odtype, output.shape)
                output_image = output_data[0]
            else:
                output_data = shm.get_contents_as_numpy(shm_op_handle, odtype, output.shape)
                output_image = output_data[0].copy()

            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, use_cuda)
            self.destroy_shm(triton_client, output_shm_name, shm_op_handle, use_cuda)
        else:
            inputs = [grpcclient.InferInput(input_name, input_data.shape, dtype)]
            inputs[0].set_data_from_numpy(input_data)
            outputs = [grpcclient.InferRequestedOutput(output_name)]
            results = triton_client.infer(model_name, inputs, outputs=outputs)
            output_image = results.as_numpy(output_name)[0]

        return output_image

    def infer2(self, img, model_name, output_byte_size,
               model_name2, output_byte_size2,
               input_name='input_2', idtype=np.uint8,
               output_name='tf.cast_1', shm_mode=1):
        input_data = np.expand_dims(img, axis=0)
        if input_data.dtype != idtype:
            input_data = input_data.astype(idtype)
        dtype = np_to_triton_dtype(idtype)

        triton_client = grpcclient.InferenceServerClient(url='localhost:8001')

        use_cuda = shm_mode > 1
        input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
        input_byte_size = input_data.nbytes
        shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, use_cuda)

        output_shm_name = "output-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
        shm_op_handle = self.create_shm(triton_client, output_shm_name, output_byte_size, use_cuda)

        if use_cuda:
            cudashm.set_shared_memory_region(shm_ip_handle, [input_data])
        else:
            shm.set_shared_memory_region(shm_ip_handle, [input_data])
        inputs = [grpcclient.InferInput(input_name, input_data.shape, dtype)]
        inputs[-1].set_shared_memory(input_shm_name, input_byte_size)
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        outputs[-1].set_shared_memory(output_shm_name, output_byte_size)
        results = triton_client.infer(model_name, inputs, outputs=outputs)
        output = results.get_output(output_name)

        inputs = [grpcclient.InferInput(input_name, output.shape, 'UINT8')]
        inputs[-1].set_shared_memory(output_shm_name, output_byte_size)
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        outputs[-1].set_shared_memory(input_shm_name, output_byte_size2)
        results = triton_client.infer(model_name2, inputs, outputs=outputs)
        output = results.get_output(output_name)
        if use_cuda:
            output_data = cudashm.get_contents_as_numpy(shm_ip_handle, np.uint8, output.shape)
            output_image = output_data[0]
        else:
            output_data = shm.get_contents_as_numpy(shm_ip_handle, np.uint8, output.shape)
            output_image = output_data[0].copy()

        self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, use_cuda)
        self.destroy_shm(triton_client, output_shm_name, shm_op_handle, use_cuda)

        return output_image

    def infer_edge(self, img, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        batched_image_data = np.expand_dims(img, axis=0)
        batched_image_data = batched_image_data.astype(np.float32)
        inputs = [grpcclient.InferInput('input_1', batched_image_data.shape, 'FP32')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = batched_image_data.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [batched_image_data])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [batched_image_data])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(batched_image_data)
        outputs = [grpcclient.InferRequestedOutput('tf_op_layer_vertices')]
        results = triton_client.infer('edge_model', inputs, outputs=outputs)
        vertices = results.as_numpy('tf_op_layer_vertices').astype('float')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return vertices

    def infer_det(self, img, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        inputs = [grpcclient.InferInput('image', img.shape, 'UINT8')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = img.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [img])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [img])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(img)
        outputs = [grpcclient.InferRequestedOutput('yyt_inclined_boxes'),
                   grpcclient.InferRequestedOutput('yyt_inclined_classes'),
                   grpcclient.InferRequestedOutput('yyt_inclined_scores'),
                   grpcclient.InferRequestedOutput('calc_inclined_boxes'),
                   grpcclient.InferRequestedOutput('calc_inclined_classes'),
                   grpcclient.InferRequestedOutput('calc_inclined_scores')]
        results = triton_client.infer('det_model', inputs, outputs=outputs)
        boxes = results.as_numpy('yyt_inclined_boxes')
        clses = results.as_numpy('yyt_inclined_classes')
        scores = results.as_numpy('yyt_inclined_scores')
        calc_boxes = results.as_numpy('calc_inclined_boxes')
        calc_clses = results.as_numpy('calc_inclined_classes')
        calc_scores = results.as_numpy('calc_inclined_scores')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return boxes, scores, clses, calc_boxes, calc_scores, calc_clses
        #return boxes, scores, clses, None, None, None

    def infer_det2(self, img, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        inputs = [grpcclient.InferInput('images', img.shape, 'FP32')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = img.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [img])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [img])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(img)
        outputs = [grpcclient.InferRequestedOutput('output'),
                  ]
        results = triton_client.infer('det_model2', inputs, outputs=outputs)
        output = results.as_numpy('output')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return output


    def infer_table_det(self, img, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        inputs = [grpcclient.InferInput('image_tensor', img.shape, 'UINT8')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = img.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [img])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [img])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(img)
        outputs = [grpcclient.InferRequestedOutput('inclined_boxes'),
                   grpcclient.InferRequestedOutput('inclined_classes'),
                   grpcclient.InferRequestedOutput('inclined_scores'), ]
        results = triton_client.infer('table_det_model', inputs, outputs=outputs)
        boxes = results.as_numpy('inclined_boxes')
        clses = results.as_numpy('inclined_classes')
        scores = results.as_numpy('inclined_scores')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return boxes, scores, clses

    def infer_publay_det(self, img, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        inputs = [grpcclient.InferInput('images', img.shape, 'FP32')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = img.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [img])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [img])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(img)
        outputs = [grpcclient.InferRequestedOutput('output'),
                   ]
        results = triton_client.infer('publay_det_model', inputs, outputs=outputs)
        output = results.as_numpy('output')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return output

    def infer_table_onnx_det(self, img, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        inputs = [grpcclient.InferInput('images', img.shape, 'FP32')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = img.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [img])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [img])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(img)
        outputs = [grpcclient.InferRequestedOutput('output'),
                   ]
        results = triton_client.infer('table_onnx_det_model', inputs, outputs=outputs)
        output = results.as_numpy('output')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return output

    def infer_cell_det(self, img, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        inputs = [grpcclient.InferInput('images', img.shape, 'FP32')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = img.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [img])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [img])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(img)
        outputs = [grpcclient.InferRequestedOutput('outputs')
                   ]
        # results = triton_client.infer('table_cell_det_model', inputs, outputs=outputs, model_version="20230301")
        results = triton_client.infer('table_cell_det_model', inputs, outputs=outputs, model_version="20221018")
        dets = results.as_numpy("outputs")
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return dets

    def infer_ocr(self, imgs, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        imgs = np.array(imgs)
        inputs = [grpcclient.InferInput('images', [50, 40, 768, 3], 'UINT8')]
        if shm_mode > 0:
            input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            input_byte_size = 50 * 40 * 768 * 3#imgs.nbytes
            shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, shm_mode > 1)
            if shm_mode > 1:
                cudashm.set_shared_memory_region(shm_ip_handle, [imgs])
            else:
                shm.set_shared_memory_region(shm_ip_handle, [imgs])
            inputs[0].set_shared_memory(input_shm_name, input_byte_size)
        else:
            inputs[0].set_data_from_numpy(imgs)
        outputs = [grpcclient.InferRequestedOutput('strings')]
        results = triton_client.infer('print_model', inputs, outputs=outputs)
        results = results.as_numpy('strings')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, shm_mode > 1)
        return results[:len(imgs)]

    def infer_classify(self, input_ids, attention_mask, token_type_ids, shm_mode=1):
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
        use_cuda = shm_mode > 1
        if shm_mode > 0:
            input_byte_size = input_ids.nbytes
            input_shm_name_1 = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            shm_ip_handle_1 = self.create_shm(triton_client, input_shm_name_1, input_byte_size, use_cuda)
            inputs = []
            if use_cuda:
                cudashm.set_shared_memory_region(shm_ip_handle_1, [input_ids])
            else:
                shm.set_shared_memory_region(shm_ip_handle_1, [input_ids])
            inputs.append(grpcclient.InferInput('input_ids', input_ids.shape, 'INT32'))
            inputs[-1].set_shared_memory(input_shm_name_1, input_byte_size)

            input_byte_size = attention_mask.nbytes
            input_shm_name_2 = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            shm_ip_handle_2 = self.create_shm(triton_client, input_shm_name_2, input_byte_size, use_cuda)
            if use_cuda:
                cudashm.set_shared_memory_region(shm_ip_handle_2, [attention_mask])
            else:
                shm.set_shared_memory_region(shm_ip_handle_2, [attention_mask])
            inputs.append(grpcclient.InferInput('attention_mask', attention_mask.shape, 'INT32'))
            inputs[-1].set_shared_memory(input_shm_name_2, input_byte_size)

            input_byte_size = token_type_ids.nbytes
            input_shm_name_3 = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            shm_ip_handle_3 = self.create_shm(triton_client, input_shm_name_3, input_byte_size, use_cuda)
            if use_cuda:
                cudashm.set_shared_memory_region(shm_ip_handle_3, [token_type_ids])
            else:
                shm.set_shared_memory_region(shm_ip_handle_3, [token_type_ids])
            inputs.append(grpcclient.InferInput('token_type_ids', token_type_ids.shape, 'INT32'))
            inputs[-1].set_shared_memory(input_shm_name_3, input_byte_size)

            output_shm_name = "output-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
            output_byte_size = 11 * 4 * input_ids.shape[0]
            shm_op_handle = self.create_shm(triton_client, output_shm_name, output_byte_size, use_cuda)
            outputs = [grpcclient.InferRequestedOutput('output')]
            outputs[-1].set_shared_memory(output_shm_name, output_byte_size)

            results = triton_client.infer('classifier_model', inputs, outputs=outputs)

            output = results.get_output("output")
            if use_cuda:
                output = cudashm.get_contents_as_numpy(shm_op_handle, np.float32, output.shape)
            else:
                output = shm.get_contents_as_numpy(shm_op_handle, np.float32, output.shape).copy()
        else:
            inputs = [grpcclient.InferInput('attention_mask', input_ids.shape, 'INT32'),
                      grpcclient.InferInput('input_ids', input_ids.shape, 'INT32'),
                      grpcclient.InferInput('token_type_ids', input_ids.shape, 'INT32')]
            inputs[0].set_data_from_numpy(input_ids)
            inputs[1].set_data_from_numpy(attention_mask)
            inputs[2].set_data_from_numpy(token_type_ids)
            outputs = [grpcclient.InferRequestedOutput('output')]
            triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=False)
            results = triton_client.infer('classifier_model', inputs, outputs=outputs)
            output = results.as_numpy('output')
        if shm_mode > 0:
            self.destroy_shm(triton_client, input_shm_name_1, shm_ip_handle_1, use_cuda)
            self.destroy_shm(triton_client, input_shm_name_2, shm_ip_handle_2, use_cuda)
            self.destroy_shm(triton_client, input_shm_name_3, shm_ip_handle_3, use_cuda)
            self.destroy_shm(triton_client, output_shm_name, shm_op_handle, use_cuda)

        return output

    def infer_orie(self, imgs, model_name,
                   input_name='images', idtype=np.uint8,
                   output_name='logits', model_version=""):
        dtype = np_to_triton_dtype(idtype)
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001')
        input_shm_name = "input-data-" + str(uuid.uuid4()) + "-" + str(os.getpid())
        input_byte_size = imgs.nbytes
        shm_ip_handle = self.create_shm(triton_client, input_shm_name, input_byte_size, False)
        shm.set_shared_memory_region(shm_ip_handle, [imgs])
        inputs = [grpcclient.InferInput(input_name, imgs.shape, dtype)]
        inputs[-1].set_shared_memory(input_shm_name, input_byte_size)
        outputs = [grpcclient.InferRequestedOutput(output_name)]
        results = triton_client.infer(model_name, inputs, model_version=model_version, outputs=outputs)
        orie = results.as_numpy(output_name)
        self.destroy_shm(triton_client, input_shm_name, shm_ip_handle, False)
        return orie
