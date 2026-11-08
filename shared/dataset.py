import os
import cv2
import random
import torch
import numpy as np
import _pickle as cPickle
import torch.utils.data as data
import copy
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from shared.data_augmentation import data_augment, defor_2D, get_rotation
from shared.image_ops import (
    aug_bbox_DZI,
    crop_resize_by_warp_affine,
    get_2d_coord_np,
    get_bbox,
    load_depth,
)
from shared.pose_geometry import matrix_to_rotation_6d


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class NOCSDataSet(data.Dataset):
    def __init__(self,
                 dynamic_zoom_in_params,
                 deform_2d_params,
                 source=None,
                 mode='train',
                 data_dir=None,
                 n_pts=1024,
                 img_size=256,
                 per_obj='',
                 pointcloud_only=False,
                 ):
        '''
        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size
        self.dynamic_zoom_in_params = dynamic_zoom_in_params
        self.deform_2d_params = deform_2d_params
        self.pointcloud_only = bool(pointcloud_only)

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']

        if mode == 'train':
            del img_list_path[2:]
            del model_file_path[2:]
        else:
            del img_list_path[:2]
            del model_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            del model_file_path[0]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]

        img_list = []
        subset_len = []
        #  aggregate all availabel datasets
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        if source == 'CAMERA':
            self.id2cat_name = self.id2cat_name_CAMERA
        self.per_obj = per_obj
        self.per_obj_id = None

        # only train one object
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(data_dir, img_list_cache_filename))]
            else:
                # needs to reorganize img_list
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
                # iter over  all img_list, cal sublen

            if len(subset_len) == 2:
                camera_len  = 0
                real_len = 0
                for i in range(len(img_list_obj)):
                    if 'CAMERA' in img_list_obj[i].split('/'):
                        camera_len += 1
                    else:
                        real_len += 1
                self.subset_len = [camera_len, real_len]
            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)

        models = {}
        for path in model_file_path:
            with open(os.path.join(self.data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        # Dense (4096-point) canonical complete clouds, same frame/scale as
        # self.models. Used as the completion target (true full-shape cloud).
        dense_path = os.path.join(self.data_dir, 'obj_models/models_dense.pkl')
        self.models_dense = {}
        if os.path.exists(dense_path):
            with open(dense_path, 'rb') as f:
                self.models_dense = cPickle.load(f)
            print(f'loaded dense models: {len(self.models_dense)}')

        # move the center to the body of the mug
        # meta info for re-label mug category
        with open(os.path.join(self.data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float32)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float32)

        self.invaild_list = []
        with open(os.path.join(self.data_dir, 'Real/train/mug_handle.pkl'), 'rb') as f:
            self.mug_sym = cPickle.load(f)
        # self.mug_sym = mmcv.load(os.path.join(self.data_dir, 'Real/train/mug_handle.pkl'))

        print(
            '{} point-cloud frames indexed{}.'.format(
                self.length,
                ' (RGB disabled)' if self.pointcloud_only else '',
            )
        )
        print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Offline teacher generation may address an exact image/object pair.
        # Normal training keeps the original random-object behaviour.
        forced_instance_index = None
        deterministic_point_seed = None
        if isinstance(index, (tuple, list)):
            index_spec = index
            index, forced_instance_index = int(index_spec[0]), int(index_spec[1])
            if len(index_spec) >= 3:
                deterministic_point_seed = int(index_spec[2])
        img_path = os.path.join(self.data_dir, self.img_list[index])
        if img_path in self.invaild_list:
            return self.__getitem__((index + 1) % self.__len__())
        try:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return self.__getitem__((index + 1) % self.__len__())
        if 'CAMERA' in img_path.split('/'):
            out_camK = self.camera_intrinsics
            img_type = 'syn'
        else:
            out_camK = self.real_intrinsics
            img_type = 'real'

        # select one foreground object,
        # if specified, then select the object
        if forced_instance_index is not None:
            idx = forced_instance_index
        elif self.per_obj != '':
            idx = gts['class_ids'].index(self.per_obj_id)
        else:
            idx = random.randint(0, len(gts['instance_ids']) - 1)
            '''
            ############### remove selected categories ###############
            remove_ids = self.cat_name2id['bowl']
            idx = None
            for i in range(10):
                idx_i = random.randint(0, len(gts['instance_ids']) - 1)
                if gts['class_ids'][idx_i] != remove_ids:
                    idx = idx_i
                    break
            if idx is None:
                return self.__getitem__((index + 1) % self.__len__())
            ##########################################################
            '''

        if gts['class_ids'][idx] == 6 and img_type == 'real':
            if self.mode == 'train':
                handle_tmp_path = img_path.split('/')
                scene_label = handle_tmp_path[-2] + '_res'
                img_id = int(handle_tmp_path[-1])
                mug_handle = self.mug_sym[scene_label][img_id]
            else:
                mug_handle = gts['handle_visibility'][idx]
        else:
            mug_handle = 1

        depth_path = img_path + '_depth.png'

        if os.path.exists(depth_path):
            depth = load_depth(depth_path)
        else:
            return self.__getitem__((index + 1) % self.__len__())

        if self.pointcloud_only:
            im_H, im_W = depth.shape[:2]
        else:
            rgb = cv2.imread(img_path + '_color.png')
            if rgb is not None:
                rgb = rgb[:, :, :3]
            else:
                return self.__getitem__((index + 1) % self.__len__())
            im_H, im_W = rgb.shape[0], rgb.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)

        mask_path = img_path + '_mask.png'
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask = mask[:, :, 2]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        # coord = cv2.imread(img_path + '_coord.png')
        # if coord is not None:
        #     coord = coord[:, :, :3]
        #     pass
        # else:
        #     return self.__getitem__((index + 1) % self.__len__())

        # aggragate information about the selected object
        inst_id = gts['instance_ids'][idx]
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        # here resize and crop to a fixed size 256 x 256
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(self.dynamic_zoom_in_params, bbox_xyxy, im_H, im_W)
        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        mask_target = mask.copy().astype(np.float32)
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0

        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, self.img_size, interpolation=cv2.INTER_NEAREST
        )

        roi_depth = np.expand_dims(roi_depth, axis=0)
        # normalize depth
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())
        roi_m_d_valid = roi_mask.astype(np.bool_) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())

        # cat_id, rotation translation and scale
        cat_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
        # note that this is nocs model, normalized along diagonal axis
        model_name = gts['model_list'][idx]
        model = self.models[gts['model_list'][idx]].astype(np.float32)  # 1024 points
        nocs_scale = gts['scales'][idx]  # nocs_scale = image file / model file
        # fsnet scale (from model) scale residual
        fsnet_scale, mean_shape = self.get_fs_net_scale(self.id2cat_name[str(cat_id + 1)], model, nocs_scale)
        fsnet_scale = fsnet_scale / 1000.0
        mean_shape = mean_shape / 1000.0
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]
        # add nnoise to roi_mask

        # pcl_in = self._depth_to_pcl(roi_depth, out_camK, roi_coord_2d, roi_mask) / 1000.0
        # np.savetxt('pts.txt', pcl_in)
        roi_mask_def = defor_2D(
            roi_mask,
            rand_r=self.deform_2d_params['roi_mask_r'],
            rand_pro=self.deform_2d_params['roi_mask_pro']
        )
        pcl_in = self._depth_to_pcl(roi_depth, out_camK, roi_coord_2d, roi_mask_def) / 1000.0
        # np.savetxt('pts_def.txt', pcl_in)

        if len(pcl_in) < 50:
            return self.__getitem__((index + 1) % self.__len__())
        point_rng = (
            np.random.default_rng(deterministic_point_seed)
            if deterministic_point_seed is not None
            else None
        )
        pcl_in = self._sample_points(pcl_in, self.n_pts, rng=point_rng)
        # sym
        sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=mug_handle)
        # generate augmentation parameters
        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()

        data_dict = {}
        data_dict['pcl_in'] = torch.as_tensor(pcl_in.astype(np.float32)).contiguous()
        data_dict['cat_id'] = torch.as_tensor(cat_id, dtype=torch.int8).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous()
        data_dict['fsnet_scale'] = torch.as_tensor(fsnet_scale, dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_info.astype(np.float32)).contiguous()
        data_dict['mean_shape'] = torch.as_tensor(mean_shape, dtype=torch.float32).contiguous()
        data_dict['aug_bb'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
        data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        data_dict['model_point'] = torch.as_tensor(model, dtype=torch.float32).contiguous()
        if self.models_dense and model_name in self.models_dense:
            data_dict['model_point_dense'] = torch.as_tensor(
                self.models_dense[model_name], dtype=torch.float32
            ).contiguous()
        data_dict['nocs_scale'] = torch.as_tensor(nocs_scale, dtype=torch.float32).contiguous()
        data_dict['handle_visibility'] = torch.as_tensor(int(mug_handle), dtype=torch.int8).contiguous()
        data_dict['instance_index'] = torch.as_tensor(idx, dtype=torch.int16).contiguous()
        # Stable CAD identity is needed for category-wise unseen-shape holdouts.
        # PyTorch's default collate keeps this as a list of strings.
        data_dict['model_name'] = str(model_name)
        data_dict['path'] = img_path
        return data_dict


    def _get_depth_normalize(self, roi_depth, roi_m_d_valid):
        depth_v_value = roi_depth[roi_m_d_valid]
        depth_normalize = (roi_depth - np.min(depth_v_value)) / (np.max(depth_v_value) - np.min(depth_v_value))
        depth_normalize[~roi_m_d_valid] = 0.0
        return depth_normalize


    def _sample_points(self, pcl, n_pts, rng=None):
        """ Down sample the point cloud using farthest point sampling.

        Args:
            pcl (torch tensor or numpy array):  NumPoints x 3
            num (int): target point number
        """
        total_pts_num = pcl.shape[0]
        if total_pts_num < n_pts:
            pcl = np.concatenate([np.tile(pcl, (n_pts // total_pts_num, 1)), pcl[:n_pts % total_pts_num]], axis=0)
        elif total_pts_num > n_pts:
            permutation = rng.permutation if rng is not None else np.random.permutation
            ids = permutation(total_pts_num)[:n_pts]
            pcl = pcl[ids]
        return pcl


    def _depth_to_pcl(self, depth, K, xymap, mask):
        K = K.reshape(-1)
        cx, cy, fx, fy = K[2], K[5], K[0], K[4]
        depth = depth.reshape(-1).astype(np.float32)
        valid = ((depth > 0) * mask.reshape(-1)) > 0
        depth = depth[valid]
        x_map = xymap[0].reshape(-1)[valid]
        y_map = xymap[1].reshape(-1)[valid]
        real_x = (x_map - cx) * depth / fx
        real_y = (y_map - cy) * depth / fy
        pcl = np.stack((real_x, real_y, depth), axis=-1)
        return pcl.astype(np.float32)


    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


    def get_fs_net_scale(self, c, model, nocs_scale):
        # model pc x 3
        lx = max(model[:, 0]) - min(model[:, 0])
        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])

        # real scale
        lx_t = lx * nocs_scale * 1000
        ly_t = ly * nocs_scale * 1000
        lz_t = lz * nocs_scale * 1000

        if c == 'bottle':
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 'bowl':
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 'camera':
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 'can':
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 'laptop':
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 'mug':
            unitx = 146
            unity = 83
            unitz = 114
        elif c == '02876657':
            unitx = 324 / 4
            unity = 874 / 4
            unitz = 321 / 4
        elif c == '02880940':
            unitx = 675 / 4
            unity = 271 / 4
            unitz = 675 / 4
        elif c == '02942699':
            unitx = 464 / 4
            unity = 487 / 4
            unitz = 702 / 4
        elif c == '02946921':
            unitx = 450 / 4
            unity = 753 / 4
            unitz = 460 / 4
        elif c == '03642806':
            unitx = 581 / 4
            unity = 445 / 4
            unitz = 672 / 4
        elif c == '03797390':
            unitx = 670 / 4
            unity = 540 / 4
            unitz = 497 / 4
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
            raise NotImplementedError
        # scale residual
        return np.array([lx_t - unitx, ly_t - unity, lz_t - unitz]), np.array([unitx, unity, unitz])


    def get_sym_info(self, c, mug_handle=1):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int8)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int8)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=np.int8)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int8)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int8)
        elif c == 'mug' and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=np.int8)  # for mug, we currently mark it as no symmetry
        elif c == 'mug' and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=np.int8)
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int8)
        return sym


def get_data_loaders(
    batch_size,
    seed,
    dynamic_zoom_in_params,
    deform_2d_params,
    percentage_data=1.0,
    data_path=None,
    source='CAMERA+Real',
    mode='train',
    n_pts=1024,
    img_size=256,
    per_obj='',
    num_workers=32,
    distributed=False,
    rank=0,
    world_size=1,
    pointcloud_only=False,
    camera_real_ratio=3.0,
    persistent_workers=None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = NOCSDataSet(
        dynamic_zoom_in_params=dynamic_zoom_in_params,
        deform_2d_params=deform_2d_params,
        source=source,
        mode=mode,
        data_dir=data_path,
        n_pts=n_pts,
        img_size=img_size,
        per_obj=per_obj,
        pointcloud_only=pointcloud_only,
    )

    if mode == 'train':
        shuffle = True
        num_workers = num_workers
    else:
        shuffle = False
        num_workers = 1

    if distributed and (world_size <= 1 or rank < 0 or rank >= world_size):
        raise ValueError('distributed loader requires 0 <= rank < world_size and world_size > 1')

    if source == 'CAMERA+Real' and mode == 'train':
        # CAMERA : Real defaults to 3 : 1.  The ratio can be adjusted while
        # still traversing every CAMERA frame at least once per epoch.
        if camera_real_ratio <= 0.0:
            raise ValueError('camera_real_ratio must be positive')
        camera_len = dataset.subset_len[0]
        real_len = dataset.subset_len[1]
        if camera_len <= 0 or real_len <= 0:
            raise ValueError('CAMERA+Real training requires both subsets')
        real_indices = list(range(camera_len, camera_len+real_len))
        camera_indices = list(range(camera_len))
        target_real = max(real_len, int(round(camera_len / camera_real_ratio)))
        repeats, remainder = divmod(target_real, real_len)
        idx = camera_indices + real_indices * repeats + real_indices[:remainder]
        random.shuffle(idx)
        size = int(percentage_data * len(idx))
        idx = idx[:size]
        # ``idx`` intentionally contains repeated Real examples to preserve
        # the 3:1 CAMERA:Real curriculum.  For DDP, wrap it in a Subset so
        # DistributedSampler shards *positions in this repeated list* rather
        # than silently dropping the oversampling policy.
        if distributed:
            train_dataset = torch.utils.data.Subset(dataset, idx)
            data_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
        else:
            train_dataset = dataset
            data_sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)

        keep_workers = (
            num_workers > 0
            if persistent_workers is None
            else bool(persistent_workers and num_workers > 0)
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=data_sampler,
            num_workers=num_workers,
            persistent_workers=keep_workers,
            drop_last=False,
            pin_memory=True,
        )

    else:
        # sample
        size = int(percentage_data * len(dataset))
        dataset, _ = torch.utils.data.random_split(dataset, (size, len(dataset) - size))
        if distributed:
            data_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=False,
            )
        else:
            data_sampler = None

        # train_dataloader = torch.utils.data.DataLoader(
        keep_workers = (
            num_workers > 0
            if persistent_workers is None
            else bool(persistent_workers and num_workers > 0)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if data_sampler is None else False,
            sampler=data_sampler,
            num_workers=num_workers,
            persistent_workers=keep_workers,
            drop_last=False,
            pin_memory=True,
        )

    return dataloader


def get_data_loaders_from_cfg(cfg, data_type=['train', 'val', 'test']):
    data_loaders = {}
    if 'train' in data_type:
        train_loader = get_data_loaders(
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_train,
            data_path=cfg.data_path,
            source=cfg.train_source,
            mode='train',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['train_loader'] = train_loader

    if 'val' in data_type:
        val_loader = get_data_loaders(
            batch_size=cfg.mini_bs,
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_val,
            data_path=cfg.data_path,
            source=cfg.val_source,
            mode='test',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['val_loader'] = val_loader

    if 'test' in data_type:
        test_loader = get_data_loaders(
            batch_size=cfg.mini_bs,
            seed=cfg.seed,
            dynamic_zoom_in_params=cfg.DYNAMIC_ZOOM_IN_PARAMS,
            deform_2d_params=cfg.DEFORM_2D_PARAMS,
            percentage_data=cfg.percentage_data_for_test,
            data_path=cfg.data_path,
            source=cfg.test_source,
            mode='test',
            n_pts=cfg.num_points,
            img_size=cfg.img_size,
            per_obj=cfg.per_obj,
            num_workers=cfg.num_workers,
        )
        data_loaders['test_loader'] = test_loader

    return data_loaders


def process_batch(batch_sample,
                  device,
                  pose_mode='quat_wxyz',
                  mini_batch_size=None,
                  PTS_AUG_PARAMS=None,
                  return_full_cloud=False,
                  full_cloud_density=1,
                  full_cloud_use_dense=True):

    if pose_mode != 'rot_matrix':
        raise ValueError("the formal Stage1/Stage2 pipeline supports rot_matrix only")
    if PTS_AUG_PARAMS==None:
        PC_da = batch_sample['pcl_in'].to(device)
        gt_R_da = batch_sample['rotation'].to(device)
        gt_t_da = batch_sample['translation'].to(device)
        gt_s_da = batch_sample['fsnet_scale'].to(device)
        mean_shape_da = batch_sample['mean_shape'].to(device)
        model_point_da = batch_sample['model_point'].to(device)
        nocs_scale_da = batch_sample['nocs_scale'].to(device)
        model_point_dense = None
        if 'model_point_dense' in batch_sample:
            model_point_dense = batch_sample['model_point_dense'].to(device)
    else:
        model_point_dense = None
        if 'model_point_dense' in batch_sample:
            model_point_dense = batch_sample['model_point_dense'].to(device)
        PC_da, gt_R_da, gt_t_da, gt_s_da, model_point_da, model_point_dense = data_augment(
            pts_aug_params=PTS_AUG_PARAMS,
            PC=batch_sample['pcl_in'].to(device),
            gt_R=batch_sample['rotation'].to(device),
            gt_t=batch_sample['translation'].to(device),
            gt_s=batch_sample['fsnet_scale'].to(device),
            mean_shape=batch_sample['mean_shape'].to(device),
            sym=batch_sample['sym_info'].to(device),
            aug_bb=batch_sample['aug_bb'].to(device),
            aug_rt_t=batch_sample['aug_rt_t'].to(device),
            aug_rt_r=batch_sample['aug_rt_R'].to(device),
            model_point=batch_sample['model_point'].to(device),
            nocs_scale=batch_sample['nocs_scale'].to(device),
            obj_ids=batch_sample['cat_id'].to(device),
            return_model_point=True,
            model_point_dense=model_point_dense,
        )
        nocs_scale_da = batch_sample['nocs_scale'].to(device)
        mean_shape_da = batch_sample['mean_shape'].to(device)

    processed_sample = {}
    processed_sample['pts'] = PC_da                # [bs, 1024, 3]
    processed_sample['pts_color'] = PC_da          # [bs, 1024, 3]
    processed_sample['id'] = batch_sample['cat_id'].to(device)      # [bs]
    processed_sample['handle_visibility'] = batch_sample['handle_visibility'].to(device)     # [bs]
    # processed_sample['path'] = batch_sample['path']
    rot = matrix_to_rotation_6d(gt_R_da.permute(0, 2, 1)).reshape(
        gt_R_da.shape[0], -1
    )

    location = gt_t_da # [bs, 3]
    processed_sample['gt_pose'] = torch.cat([rot.float(), location.float()], dim=-1)   # [bs, 4/6/3 + 3]
    # Metric dimensions of the canonical 3D bounding box.  ``fsnet_scale``
    # is stored as a residual around ``mean_shape`` in metres; their sum is
    # exactly ``(model max - model min) * nocs_scale``.  The retained models
    # predict this quantity so evaluation can compute genuine 3D IoU rather
    # than reuse the detector's all-one placeholder scales.
    processed_sample['gt_size'] = (gt_s_da + mean_shape_da).float().clamp_min(1e-5)

    """ zero center """
    num_pts = processed_sample['pts'].shape[1]
    zero_mean = torch.mean(processed_sample['pts'][:, :, :3], dim=1)
    processed_sample['zero_mean_pts'] = copy.deepcopy(processed_sample['pts'])
    processed_sample['zero_mean_pts'][:, :, :3] -= zero_mean.unsqueeze(1).repeat(1, num_pts, 1)
    processed_sample['zero_mean_gt_pose'] = copy.deepcopy(processed_sample['gt_pose'])
    processed_sample['zero_mean_gt_pose'][:, -3:] -= zero_mean
    processed_sample['pts_center'] = zero_mean

    if return_full_cloud:
        # Camera-space complete cloud consistent with the augmented observation:
        # full_cam = gt_R_aug @ (model_point_aug * nocs_scale) + gt_t_aug
        # Prefer the true dense (4096-pt) full-shape cloud when available,
        # unless a downstream model deliberately requires the native 1024-pt
        # CAD density used by REAL275 test models.
        if model_point_dense is not None and full_cloud_use_dense:
            target_model = model_point_dense.float()
        else:
            target_model = model_point_da.float()
            if full_cloud_density > 1:
                # Fallback densification (1024 -> density*1024) with jitter.
                target_model = target_model.repeat_interleave(full_cloud_density, dim=1)
                target_model = target_model + torch.randn_like(target_model) * 0.002
        full_cloud = torch.matmul(
            gt_R_da.float(),
            (target_model * nocs_scale_da.float().unsqueeze(-1).unsqueeze(-1)).transpose(-1, -2),
        ).transpose(-1, -2) + gt_t_da.float().unsqueeze(-2)
        processed_sample['full_cloud'] = full_cloud  # [bs, N_full, 3] camera space
        # The complete camera-frame cloud is anchored at the observed depth
        # cloud's centroid, never at a canonical CAD origin or GT translation.
        # Stage1 uses the same center for its translation residual contract.
        processed_sample['full_cloud_alignment_center'] = zero_mean
        processed_sample['full_cloud_relative'] = full_cloud - zero_mean.unsqueeze(1)

    if 'color' in batch_sample.keys():
        pass
        # processed_sample['color'] = batch_sample['color'].to(device)       # [bs]

    if not mini_batch_size == None:
        for key in processed_sample.keys():
            processed_sample[key] = processed_sample[key][:mini_batch_size]

    if not 'color' in processed_sample.keys():
        pass
        # processed_sample['color'] = None
    # print(processed_sample['zero_mean_pts'].device)
    return processed_sample
