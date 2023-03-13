from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from pyquaternion import Quaternion
import numpy as np
import torch
import json
import cv2
import os
import math
import copy
from tqdm import tqdm
import pickle
from ..generic_dataset import GenericDataset
from utils.ddd_utils import compute_box_3d, project_to_image, iou3d_global
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from itertools import compress
from glob import glob
from pathlib import Path as P
from .. import calibration_kitti
from utils.image import get_affine_transform, affine_transform

class vod(GenericDataset):
    num_categories = 3
    default_resolution = [384, 1280]
    # ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
    #       'Tram', 'Misc', 'DontCare']
    class_name = ['Pedestrian', 'Car', 'Cyclist']
    # negative id is for "not as negative sample for abs(id)".
    # 0 for ignore losses for all categories in the bounding box region
    cat_ids = {1:1, 2:2, 3:3, 4:-2, 5:-2, 6:-1, 7:-9999, 8:-9999, 9:0}
    max_objs = 50
    
    
    def __init__(self, opt, split):
        '''
        opt.data_dir = '~data/
        '''
        self._data_rng = np.random.RandomState(123) # for color aug
        self.split = split
        self.enable_meta = True if (opt.run_dataset_eval and split in ["val", "mini_val", "test"]) or opt.eval else False
        data_dir = os.path.join(opt.data_dir, 'vod')
        # img_dir = os.path.join(data_dir, 'training', 'trainval')
        # if opt.trainval:
        split = 'testing' if self.split == 'test' else 'training'
        self.split_path = P(os.path.join(data_dir, 'radar', split))
        img_dir = os.path.join(data_dir, 'radar', split, 'image_2')
        # ann_path = os.path.join(
        # data_dir, 'annotations', 'kitti_v2_{}.json').format(split)
        pickle_name = self.split if self.split == 'val' else split[:-3]
        pickle_path = os.path.join(data_dir, 'radar', 'kitti_infos_%s.pkl'%pickle_name)
        self.coco = None
        self.img_dir = P(img_dir)
        self.opt = opt
        with open(pickle_path, 'rb') as f:
            self.info = pickle.load(f)
        # self.images = 
        # else:
        #     ann_path = os.path.join(data_dir, 
        #     'annotations', 'kitti_v2_{}_{}.json').format(opt.kitti_split, split)

        # self.images = None
        # # load image list and coco
        # super(KITTI, self).__init__(opt, split, ann_path, img_dir)
        # self.alpha_in_degree = False
        # self.num_samples = len(self.images)

        # print('Loaded {} {} samples'.format(split, self.num_samples))
        
    def __len__(self):
        return len(self.info)

    def get_calib(self, idx):
        calib_file = self.split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)
    
    @staticmethod
    def get_coco_yaw(yaw_kitti):
        # Compute the rotation matrix that rotates the object's longitudinal axis by the yaw angle
        R = np.array([[np.cos(yaw_kitti), -np.sin(yaw_kitti), 0],
                    [np.sin(yaw_kitti), np.cos(yaw_kitti), 0],
                    [0, 0, 1]])

        # Rotate the object's longitudinal axis by the yaw angle to obtain the orientation in the global coordinate system
        long_axis_global = np.dot(R, np.array([1, 0, 0]))

        # Compute the yaw angle in the COCO convention
        yaw_coco = np.arctan2(-long_axis_global[0], long_axis_global[2])
        return yaw_coco

    @staticmethod
    def get_coco_bbox(bboxes):
        bbox_num = bboxes.shape[0]
        coco_bboxes = np.zeros_like(bboxes)
        for i in range(bbox_num):
            bbox = bboxes[i,:]
            coco_bboxes[i,:] = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        return coco_bboxes
    
    @staticmethod
    def keep_arrays_by_name(gt_names, used_classes):
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds
    
    
    @staticmethod
    def get_amodel(location, kitti_yaw, dims):

        # Extract the location and yaw information from the KITTI bounding box
        x, y, z = location
        yaw = kitti_yaw

        # Convert the yaw to the rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([[cos_yaw, 0, -sin_yaw], [0, 1, 0], [sin_yaw, 0, cos_yaw]])

        # Convert the location to the amodel_center format
        amodel_center = np.dot(rotation_matrix.T, np.array([x, y, z])) - np.dot(rotation_matrix.T, dims) / 2
        return amodel_center

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    @staticmethod
    def get_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    @staticmethod
    def cart2sphere(coords):
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        return np.column_stack((r, theta, phi))
    
    def get_radar(self, idx):
        radar_file = self.split_path / 'velodyne' / ('%s.bin' % idx)
        # print(radar_file)
        assert radar_file.exists()
        radar_point_cloud = np.fromfile(str(radar_file), dtype=np.float32).reshape(-1, 7)
        # if self.block_point_cloud_features:
        #     radar_point_cloud[:, 3:] = 0
        # print('loading radar point cloud')
        return radar_point_cloud
    
    def _load_pc_data(self, img, inp_trans, out_trans, calib, idx, flipped=0):
        '''
        img_info = {
            'calib':3x4 matrix
        }
        '''
        img_height, img_width = img.shape[0], img.shape[1]
        radar_pc = self.get_radar(idx)

        sphere_coord = self.cart2sphere(radar_pc[:,:3]).T
        # calculate distance to points
        depth = sphere_coord[2, :]
        
        # filter points by distance
        if self.opt.max_pc_dist > 0:
            mask = (depth <= self.opt.max_pc_dist)
            radar_pc = radar_pc[mask,:]
            depth = depth[mask]

        # add z offset to radar points
        if self.opt.pc_z_offset != 0:
            radar_pc[:, 2] -= self.opt.pc_z_offset
        
        # map points to the image and filter ones outside
        # pc_2d, mask = map_pointcloud_to_image(radar_pc, np.array(img_info['camera_intrinsic']), 
        #                         img_shape=(img_info['width'],img_info['height']))
        # pc_3d = radar_pc[:,mask]
        pts_rect = calib.lidar_to_rect(radar_pc[:, 0:3])
        fov_flag = self.get_fov_flag(pts_rect, [img_height, img_width], calib)
        radar_3d = radar_pc[fov_flag]
        pc_3d = radar_3d.T
        pc_2d = sphere_coord[:, fov_flag]
        # sort points by distance
        ind = np.argsort(pc_2d[2,:])
        pc_2d = pc_2d[:,ind]
        pc_3d = pc_3d[:,ind]

        # flip points if image is flipped
        if flipped:
            pc_2d = self._flip_pc(pc_2d,  img_width)
            pc_3d[0,:] *= -1  # flipping the x dimension
            pc_3d[8,:] *= -1  # flipping x velocity (x is right, z is front)
        calib_mat = {
            'calib':calib.P2
        }
        pc_2d, pc_3d, pc_dep = self._process_pc(pc_2d, pc_3d, img, inp_trans, out_trans, calib_mat)
        pc_N = np.array(pc_2d.shape[1])

        # pad point clouds with zero to avoid size mismatch error in dataloader
        n_points = min(self.opt.max_pc, pc_2d.shape[1])
        pc_z = np.zeros((pc_2d.shape[0], self.opt.max_pc))
        pc_z[:, :n_points] = pc_2d[:, :n_points]
        pc_3dz = np.zeros((pc_3d.shape[0], self.opt.max_pc))
        pc_3dz[:, :n_points] = pc_3d[:, :n_points]

        return pc_z, pc_N, pc_dep, pc_3dz

    
    def __getitem__(self, index):
        '''
        ret:
            'image': image
            'pc_2d': point cloud with ,
            'pc_3d':
            'pc_N': 
            'pc_dep':
            'hm': 
            'ind':
            'cat'
            'mask'
            'pc_hm'
            'reg'
            'reg_mask'
            'wh'
            'wh_mask'
            'nuscenes_att'
            'nuscenes_att_mask'
            'velocity':ignore
            'velocity_mask':ignore
            'dep'
            'dep_mask'
            'dim'
            'dim_mask'
            'amodel_offset': ignore
            'amodel_offset_mask': ignore
            'rotbin'
            'rotres'
            'rot_mask'
            'calib'
        '''
        info = self.info[index]
        idx = info['image']['image_idx']
        height, width = info['image']['image_shape']
        img_fname = idx + '.jpg'
        img_path = str(self.img_dir/img_fname)
        img = cv2.imread(img_path)
        
        # img_info, anns, img_path
        calib = self.get_calib(idx)
        
        # convert kitti bbox format to coco bbox format
        
        # ann['bbox'] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] kitti to coco
        # ann['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # ann['alpha'] = alpha
        # convert yaw from kitti to coco 
        annos = info['annos']
        selected_name = annos['name']
        selected = self.keep_arrays_by_name(selected_name, self.class_name)
        selected_annos = {}
        for k in annos.keys():
            selected_annos[k] = annos[k][selected]

        bbox = selected_annos['bbox']
        alpha = selected_annos['alpha']
        dimensions = selected_annos['dimensions']
        yaw = selected_annos['rotation_y']
        location = selected_annos['location']
        name = selected_annos['name']
        obj_id =  np.array([self.class_name.index(n) + 1 for n in name], dtype=np.int32)
        coco_bbox = self.get_coco_bbox(bbox)
        coco_yaw = [self.get_coco_yaw(y) for y in yaw]
        anns = []
        anns_num = len(obj_id)
        for i in range(anns_num):
            amodel = self.get_amodel(location[i], yaw[i], dimensions[i])
            
            ann_dict = {
                'id':0,
                'image_id':idx,
                'category_id':obj_id[i],
                'dim':dimensions[i],
                'location':location[i],
                'depth':location[i][2],
                'occluded':selected_annos['occluded'][i],
                'truncated':selected_annos['truncated'][i],
                'rotation_y':coco_yaw[i],
                'amodel_center':amodel,
                'iscrowd':0,
                'track_id':0,
                'attributes':0,
                'velocity':0,
                'velocity_cam':0,
                'bbox':coco_bbox[i],
                'area':self.get_area(coco_bbox[i]),
                'alpha':alpha[i]
            }
            anns += [ann_dict]
            
        ## Get center and scale from image
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        # if not self.opt.not_max_crop:
        #     s = max(img.shape[0], img.shape[1]) * 1.0 
        # else: 
        s = np.array([img.shape[1], img.shape[0]], np.float32)
        aug_s, rot, flipped = 1, 0, 0

        ## data augmentation for training set
        if 'train' in self.split:
            c, aug_s, rot = c, 1, 0
            # c, aug_s, rot = self._get_aug_param(c, s, width, height)
            # flip only
            # s = s * aug_s
            if np.random.random() < self.opt.flip:
                flipped = 1
                img = img[:, ::-1, :]
                anns = self._flip_anns(anns, width)
        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_w, self.opt.input_h])
        trans_output = get_affine_transform(
            c, s, rot, [self.opt.output_w, self.opt.output_h])
        inp = self._get_input(img, trans_input)
        ret = {'image': inp}
        gt_det = {'bboxes': [], 'scores': [], 'clses': [], 'cts': []}
        img_info = {
            'calib':calib.P2
        }
        if self.opt.pointcloud:
            pc_2d, pc_N, pc_dep, pc_3d = self._load_pc_data(img, trans_input, trans_output, calib, idx, flipped=0)
            ret.update({ 'pc_2d': pc_2d,
                'pc_3d': pc_3d,
                'pc_N': pc_N,
                'pc_dep': pc_dep })
            
        pre_cts, track_ids = None, None
        if self.opt.tracking:
            raise NotImplementedError('tracking not implemented for VOD yet')
        ### init samples
        if type(gt_det['bboxes']) != list:
            print('sth is wrong here')
        self._init_ret(ret, gt_det)
        velocity_mat = np.eye(4)
        num_objs = min(len(anns), self.max_objs)
        if type(gt_det['bboxes']) != list:
            print('sth is wrong here')
        for k in range(num_objs):
            if type(gt_det['bboxes']) != list:
                print('sth is wrong here')
            ann = anns[k]
            cls_id = int(self.cat_ids[ann['category_id']])
            if cls_id > self.opt.num_classes or cls_id <= -999:
                continue
            bbox, bbox_amodal = self._get_bbox_output(
                ann['bbox'], trans_output, height, width)
            if cls_id <= 0 or ('iscrowd' in ann and ann['iscrowd'] > 0):
                self._mask_ignore_or_crowd(ret, cls_id, bbox)
                continue
            if type(gt_det['bboxes']) != list:
                print('sth is wrong here')
            self._add_instance(
                ret, gt_det, k, cls_id, bbox, bbox_amodal, ann, trans_output, aug_s, 
                calib.P2, pre_cts, track_ids)
            if type(gt_det['bboxes']) != list:
                print('sth is wrong here')

        if self.opt.debug > 0 or self.enable_meta:
            gt_det = self._format_gt_det(gt_det)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': idx,
                    'img_path': img_path, 'calib': calib.P2,
                    'img_width': width, 'img_height': height,
                    'flipped': flipped, 'velocity_mat':velocity_mat}
            ret['meta'] = meta
        ret['calib'] = calib.P2
        return ret
    
    def save_results(self, results, save_dir, task, split):
        results_dir = os.path.join(save_dir, task + '_' + split + '_' + 'results_kitti')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        for img_id in results.keys():
            out_path = os.path.join(results_dir, '{:06d}.txt'.format(int(img_id)))
            f = open(out_path, 'w')
        for i in range(len(results[img_id])):
            item = results[img_id][i]
            category_id = item['class']
            cls_name_ind = category_id
            class_name = self.class_name[cls_name_ind - 1]
            if not ('alpha' in item):
                item['alpha'] = -1
            if not ('rot_y' in item):
                item['rot_y'] = -1
            if 'dim' in item:
                item['dim'] = [max(item['dim'][0], 0.01), 
                max(item['dim'][1], 0.01), max(item['dim'][2], 0.01)]
            if not ('dim' in item):
                item['dim'] = [-1000, -1000, -1000]
            if not ('loc' in item):
                item['loc'] = [-1000, -1000, -1000]
            f.write('{} 0.0 0'.format(class_name))
            f.write(' {:.2f}'.format(item['alpha']))
            f.write(' {:.2f} {:.2f} {:.2f} {:.2f}'.format(
            item['bbox'][0], item['bbox'][1], item['bbox'][2], item['bbox'][3]))
            
            f.write(' {:.2f} {:.2f} {:.2f}'.format(
            item['dim'][0], item['dim'][1], item['dim'][2]))
            f.write(' {:.2f} {:.2f} {:.2f}'.format(
            item['loc'][0], item['loc'][1], item['loc'][2]))
            f.write(' {:.2f} {:.2f}\n'.format(item['rot_y'], item['score']))
        f.close()
    
    def run_eval(self, results, save_dir, n_plots=10, render_curves=False):
        split = 'test' if self.split == 'test' else 'val'
        task = 'tracking' if self.opt.tracking else 'det'
        self.save_results(results, save_dir, task, split)
        