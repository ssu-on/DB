# #!python3
# import argparse
# import os
# import torch
# import cv2
# import numpy as np
# from experiment import Structure, Experiment
# from concern.config import Configurable, Config
# import math
# import time

# def main():
#     parser = argparse.ArgumentParser(description='Text Recognition Training')
#     parser.add_argument('exp', type=str)
#     parser.add_argument('--resume', type=str, help='Resume from checkpoint')
#     parser.add_argument('--image_path', type=str, help='image path')
#     parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
#     parser.add_argument('--data', type=str,
#                         help='The name of dataloader which will be evaluated on.')
#     parser.add_argument('--image_short_side', type=int, default=736,
#                         help='The threshold to replace it in the representers')
#     parser.add_argument('--thresh', type=float,
#                         help='The threshold to replace it in the representers')
#     parser.add_argument('--box_thresh', type=float, default=0.6,
#                         help='The threshold to replace it in the representers')
#     parser.add_argument('--visualize', action='store_true',
#                         help='visualize maps in tensorboard')
#     parser.add_argument('--resize', action='store_true',
#                         help='resize')
#     parser.add_argument('--polygon', action='store_true',
#                         help='output polygons if true')
#     parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
#                         help='Show iamges eagerly')
#     parser.add_argument('--dest', type=str, choices=['binary', 'color_embedding', 'subtitle_binary'],
#                         help='Select prediction map to feed into representer')
#     parser.add_argument('--save_binary_mask', action='store_true',
#                         help='Save thresholded binary mask as an image during inference')

#     args = parser.parse_args()
#     args = vars(args)
#     args = {k: v for k, v in args.items() if v is not None}

#     conf = Config()
#     experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
#     experiment_args.update(cmd=args)
#     experiment = Configurable.construct_class_from_config(experiment_args)

#     Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])


# class Demo:
#     def __init__(self, experiment, args, cmd=dict()):
#         self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
#         self.experiment = experiment                                            # experiment mode로 load
#         experiment.load('evaluation', **args)
#         self.args = cmd                                                         # model path, config
#         model_saver = experiment.train.model_saver
#         self.structure = experiment.structure                                           
#         self.model_path = self.args['resume']
#         self._result_dir = self.args.get('result_dir', './demo_results/')

#     def init_torch_tensor(self):
#         # Use gpu or not
#         torch.set_default_tensor_type('torch.FloatTensor')
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#             torch.set_default_tensor_type('torch.cuda.FloatTensor')
#         else:
#             self.device = torch.device('cpu')

#     def init_model(self):
#         model = self.structure.builder.build(self.device)
#         return model

#     def resume(self, model, path):
#         if not os.path.exists(path):
#             print("Checkpoint not found: " + path)
#             return
#         print("Resuming from " + path)
#         states = torch.load(
#             path, map_location=self.device)
#         model.load_state_dict(states, strict=False)
#         print("Resumed from " + path)

#     def resize_image(self, img):
#         height, width, _ = img.shape
#         if height < width:
#             new_height = self.args['image_short_side']
#             new_width = int(math.ceil(new_height / height * width / 32) * 32)
#         else:
#             new_width = self.args['image_short_side']
#             new_height = int(math.ceil(new_width / width * height / 32) * 32)
#         resized_img = cv2.resize(img, (new_width, new_height))
#         return resized_img
        
#     def load_image(self, image_path):
#         img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
#         original_shape = img.shape[:2]
#         img = self.resize_image(img)
#         img -= self.RGB_MEAN
#         img /= 255.
#         img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
#         return img, original_shape
        
#     # inference result -> txt file
#     def format_output(self, batch, output):
#         batch_boxes, batch_scores = output
#         for index in range(batch['image'].size(0)):
#             original_shape = batch['shape'][index]
#             filename = batch['filename'][index]
#             result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
#             result_file_path = os.path.join(self.args['result_dir'], result_file_name)
#             boxes = batch_boxes[index]
#             scores = batch_scores[index]
#             if self.args['polygon']:                                    # polygon의 모든 꼭짓점 좌표 저장
#                 with open(result_file_path, 'wt') as res:
#                     for i, box in enumerate(boxes):
#                         box = np.array(box).reshape(-1).tolist()
#                         result = ",".join([str(int(x)) for x in box])
#                         score = scores[i]
#                         res.write(result + ',' + str(score) + "\n")
#             else:
#                 with open(result_file_path, 'wt') as res:
#                     for i in range(boxes.shape[0]):
#                         score = scores[i]
#                         if score < self.args['box_thresh']:
#                             continue
#                         box = boxes[i,:,:].reshape(-1).tolist()
#                         result = ",".join([str(int(x)) for x in box])
#                         res.write(result + ',' + str(score) + "\n")

#     def inference(self, image_path, visualize=False):
#         self.init_torch_tensor()                                                # GPU 사용 여부 설정            
#         model = self.init_model()                                               # initialize model            
#         self.resume(model, self.model_path)                                     # load model    
#         all_matircs = {}                                                        # metrics 저장  @@    
#         model.eval()                                                            # set model to evaluation mode    
#         batch = dict()                                                          
#         batch['filename'] = [image_path]
#         img, original_shape = self.load_image(image_path)                       # load image and original shape
#         batch['shape'] = [original_shape]
#         batch['image'] = img

#         # ---------------------------------------------
#         # # warm-up
#         for _ in range(20):
#             with torch.no_grad():
#                 _ = model.forward(batch, training=False)
        
#         # ---------------------------------------------
#         # estimate FPS
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         start_time = time.time()
#         # ---------------------------------------------

#         with torch.no_grad():                                               
#             # @@ batch['image'] = img
#             pred = model.forward(batch, training=False)                         # train을 통해 binary, thresh map을 얻음
#             output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])  # post-processing

#         # ---------------------------------------------
#         torch.cuda.synchronize() if torch.cuda.is_available() else None
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         fps = 1.0 / elapsed_time
#         print(f"[INFO] Inference Time: {elapsed_time * 1000:.2f} ms  ({fps:.2f} FPS)")
#         # ---------------------------------------------
        
#         if not os.path.isdir(self.args['result_dir']):
#             os.mkdir(self.args['result_dir'])
#         if self.args.get('save_binary_mask'):
#             self.save_binary_mask(pred, batch)
#         self.format_output(batch, output)

#         if visualize and self.structure.visualizer:
#             vis_image = self.structure.visualizer.demo_visualize(image_path, output)
#             cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)

#     def save_binary_mask(self, prediction, batch):
#         """
#         Save binarized probability map (binary map) as 0/255 mask image.
#         """
#         if isinstance(prediction, dict):
#             dest_key = self.args.get('dest', self.structure.representer.dest)
#             prob_map = prediction.get(dest_key) or prediction.get('binary')
#             if prob_map is None:
#                 print(f"[WARN] No '{dest_key}' map found in prediction; skip mask saving.")
#                 return
#         else:
#             prob_map = prediction

#         thresh = self.structure.representer.thresh
#         mask_batch = (prob_map > thresh).to(torch.uint8)
#         filenames = batch['filename']
#         os.makedirs(self._result_dir, exist_ok=True)

#         for mask_tensor, file_path in zip(mask_batch, filenames):
#             if mask_tensor.ndim != 3 or mask_tensor.size(0) != 1:
#                 print("[WARN] Unexpected mask tensor shape; skip mask saving.")
#                 continue
#             mask_np = mask_tensor[0].detach().cpu().numpy().astype(np.uint8) * 255
#             save_name = os.path.splitext(os.path.basename(file_path))[0] + '_mask.png'
#             save_path = os.path.join(self._result_dir, save_name)
#             cv2.imwrite(save_path, mask_np)
#             print(f"[INFO] Saved binary mask to {save_path}")

# if __name__ == '__main__':
#     main()

#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
import time

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')
    parser.add_argument('--dest', type=str, choices=['binary', 'color_embedding', 'subtitle_binary'],
                        help='Select prediction map to feed into representer')
    parser.add_argument('--save_binary_mask', action='store_true',
                        help='Save thresholded binary mask as an image during inference')
    parser.add_argument('--mask_source', type=str, choices=['prob', 'boxes'], default='prob',
                        help='Select mask source: probability map or detected boxes')
    parser.add_argument('--mask_dilate_kernel', type=int, default=0,
                        help='Odd kernel size (>0) for dilating saved binary mask; 0 disables dilation')
    parser.add_argument('--mask_dilate_iter', type=int, default=1,
                        help='Number of dilation iterations when saving binary mask')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment                                            # experiment mode로 load
        experiment.load('evaluation', **args)
        self.args = cmd                                                         # model path, config
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure                                           
        self.model_path = self.args['resume']
        self._result_dir = self.args.get('result_dir', './demo_results/')

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    # inference result -> txt file
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:                                    # polygon의 모든 꼭짓점 좌표 저장
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def inference(self, image_path, visualize=False):
        self.init_torch_tensor()                                                # GPU 사용 여부 설정            
        model = self.init_model()                                               # initialize model            
        self.resume(model, self.model_path)                                     # load model    
        all_matircs = {}                                                        # metrics 저장  @@    
        model.eval()                                                            # set model to evaluation mode    
        batch = dict()                                                          
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)                       # load image and original shape
        batch['shape'] = [original_shape]
        batch['image'] = img

        # ---------------------------------------------
        # # warm-up
        for _ in range(20):
            with torch.no_grad():
                _ = model.forward(batch, training=False)
        
        # ---------------------------------------------
        # estimate FPS
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        # ---------------------------------------------

        with torch.no_grad():                                               
            # @@ batch['image'] = img
            pred = model.forward(batch, training=False)                         # train을 통해 binary, thresh map을 얻음
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])  # post-processing

        # ---------------------------------------------
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1.0 / elapsed_time
        print(f"[INFO] Inference Time: {elapsed_time * 1000:.2f} ms  ({fps:.2f} FPS)")
        # ---------------------------------------------
        
        if not os.path.isdir(self.args['result_dir']):
            os.mkdir(self.args['result_dir'])
        self.format_output(batch, output)
        if self.args.get('save_binary_mask'):
            self.save_binary_mask(pred, batch, output)

        if visualize and self.structure.visualizer:
            vis_image = self.structure.visualizer.demo_visualize(image_path, output)
            cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)

    def save_binary_mask(self, prediction, batch, output):
        """
        Save mask image using probability map or detected boxes.
        """
        mask_source = self.args.get('mask_source', 'prob')
        os.makedirs(self._result_dir, exist_ok=True)
        if mask_source == 'boxes':
            mask_items = self._build_box_masks(batch, output)
        else:
            mask_items = self._build_prob_masks(prediction, batch)
        if not mask_items:
            return
        kernel, dilate_iter = self._prepare_dilate_kernel()
        for mask_np, file_path in mask_items:
            if kernel is not None:
                mask_np = cv2.dilate(mask_np, kernel, iterations=dilate_iter)
            save_name = os.path.splitext(os.path.basename(file_path))[0] + '_mask.png'
            save_path = os.path.join(self._result_dir, save_name)
            cv2.imwrite(save_path, mask_np)
            print(f"[INFO] Saved binary mask to {save_path}")

    def _build_prob_masks(self, prediction, batch):
        if isinstance(prediction, dict):
            dest_key = self.args.get('dest', self.structure.representer.dest)
            prob_map = prediction.get(dest_key) or prediction.get('binary')
            if prob_map is None:
                print(f"[WARN] No '{dest_key}' map found in prediction; skip mask saving.")
                return []
        else:
            prob_map = prediction

        thresh = self.structure.representer.thresh
        mask_batch = (prob_map > thresh).to(torch.uint8)
        filenames = batch['filename']
        items = []
        for mask_tensor, file_path in zip(mask_batch, filenames):
            if mask_tensor.ndim != 3 or mask_tensor.size(0) != 1:
                print("[WARN] Unexpected mask tensor shape; skip mask saving.")
                continue
            mask_np = mask_tensor[0].detach().cpu().numpy().astype(np.uint8) * 255
            items.append((mask_np, file_path))
        return items

    def _build_box_masks(self, batch, output):
        if output is None:
            print("[WARN] No detection output available; skip box-based mask saving.")
            return []
        boxes_batch, scores_batch = output
        filenames = batch['filename']
        shapes = batch['shape']
        items = []
        box_thresh = self.args.get('box_thresh', self.structure.representer.box_thresh)
        for boxes, scores, file_path, shape in zip(boxes_batch, scores_batch, filenames, shapes):
            if isinstance(shape, torch.Tensor):
                height, width = map(int, shape.tolist())
            else:
                height, width = map(int, shape)
            mask = np.zeros((height, width), dtype=np.uint8)
            for box, score in zip(boxes, scores):
                if score < box_thresh:
                    continue
                pts = np.array(box, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
            items.append((mask, file_path))
        return items

    def _prepare_dilate_kernel(self):
        kernel_size = int(self.args.get('mask_dilate_kernel', 0))
        if kernel_size > 0:
            if kernel_size % 2 == 0:
                print("[WARN] mask_dilate_kernel must be odd; skipping dilation.")
                return None, 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            dilate_iter = max(1, int(self.args.get('mask_dilate_iter', 1)))
            return kernel, dilate_iter
        return None, 1

if __name__ == '__main__':
    main()
