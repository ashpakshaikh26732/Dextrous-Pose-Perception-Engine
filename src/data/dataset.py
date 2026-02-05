import torch 
import os 
import json 
import torchvision
import numpy as np
from scipy.spatial.transform import Rotation as R

class ycb_dataset (torch.utils.data.Dataset) : 
    def __init__(self , config : dict , root : str , is_training : bool = True , model_stage : int = 1  ) : 
        self.config = config 
        self.root = root
        self.is_training = is_training 
        self.all_samples = []

        print('starting file paths flatning ')
        self._flatten_data_paths(self.root) 
        print('completed file paths flatning ')

        self.model_stage = model_stage
        
    def _flatten_data_paths(self , root ) : 

        for vidio in sorted(os.listdir(root)): 
            vidio_address = os.path.join(root, vidio) 


        
            scene_camera_json_path = os.path.join(vidio_address ,'scene_camera.json' )
            with open(scene_camera_json_path , 'r') as jsb : 
                scene_camera_json_data = json.load(jsb) 
                
            scene_gt_json_path = os.path.join(vidio_address , 'scene_gt.json') 
            with open(scene_gt_json_path , 'r') as jsb : 
                scene_gt_json_data = json.load(jsb) 

            depth_map_dir_path = os.path.join(vidio_address , 'depth')
            rgb_image_dir_path = os.path.join(vidio_address , 'rgb')

            assert len(os.listdir(depth_map_dir_path)) == len(os.listdir(rgb_image_dir_path))

            all_rgbs = sorted(os.listdir(rgb_image_dir_path))
            all_depths= sorted(os.listdir(depth_map_dir_path))
            
            for index in range(len(all_rgbs)) : 
                obj = {}
                rgb_image = all_rgbs[index]
                depth_image = all_depths[index]
                
                assert os.path.splitext(rgb_image)[0] == os.path.splitext(depth_image)[0]

                # rgb_image_address = os.path.join(rgb_image_dir_path, rgb_image)
                # depth_map_image_address = os.path.join(depth_map_dir_path , depth_image )
                rgb_image_address = os.path.join(vidio , os.path.join('rgb', rgb_image))
                depth_map_image_address = os.path.join(vidio , os.path.join('depth' , depth_image))
                
                obj['rgb_image_adderss'] = rgb_image_address 
                obj['depth_map_image_address'] = depth_map_image_address 

                camera_settings=scene_camera_json_data[str(index)]
                obj['depth_scale'] = camera_settings['depth_scale']
                obj['cam_k'] = camera_settings['cam_K']

                obj['laebels'] = [] 

                mask_visb_path = os.path.join(vidio_address , 'mask_visib' )

                # print(f'rgb_image {rgb_image}')
                # print(f'depth_image {depth_image}')
                # print(f'full image address :- {rgb_image_address}')
                
                # total_masks = sorted([
                #     mask
                #     for mask in os.listdir(mask_visb_path)
                #     if os.path.splitext(mask)[0].split('_')[0]
                #        == os.path.splitext(rgb_image)[0]
                # ])

                scene_gt = scene_gt_json_data[str(index)]

                

                for obj_idx , mask_data in enumerate(  scene_gt) : 
                    mask_meta = {} 
                    # mask_meta['mask_address'] = os.path.join(mask_visb_path , mask)
                    
                    mask_filename = f"{int(index):06d}_{int(obj_idx):06d}.png"
                    
                    mask_meta['mask_address'] = os.path.join(vidio , os.path.join('mask_visib' , mask_filename ))
                    mask_meta['object_id'] = mask_data['obj_id'] 
                    mask_meta['cam_R_m2c'] = mask_data['cam_R_m2c'] 
                    mask_meta['cam_t_m2c'] = mask_data['cam_t_m2c'] 

                    obj['laebels'].append(mask_meta)
                self.all_samples.append(obj)

            print(f'completed {vidio}')
        

    def __len__(self) : 
        return len(self.all_samples)
    
    def _read_and_normalize_rgb_image(self, image_path : str) -> torch.Tensor : 
        full_image_path =  os.path.join(self.root, image_path)
        img  = torchvision.io.read_image(full_image_path) 
        return img / 255.0 
    
    def _read_and_standardized_depth_img(self , depth_map_path : str , depth_scale : float ) -> torch.Tensor : 
        full_depth_map_path = os.path.join(self.root , depth_map_path)
        depth_map = torchvision.io.read_image(full_depth_map_path) 
        depth_map = depth_map * depth_scale 
        return depth_map

    def process_pose(self, rot_list , trans_list) : 
        t = np.array(trans_list , dtype=np.float32) 
        t = t /1000.0 

        r_matrix = np.array(rot_list , dtype=np.float32).reshape(3,3) 

        r_quat = R.from_matrix(r_matrix).as_quat() 
        
        t_tensor  = torch.from_numpy(t).float() 
        r_tensor = torch.from_numpy(r_quat)

        return t_tensor , r_tensor 
    

    def _read_masks_and_other_meta_data(self , labels : list[dict]) : 
        mask_lists = [] 
        cam_r_m2c_list = [] 
        cam_t_m2c_list = [] 
        object_id_list = [] 

        for item in labels : 
            img_path = item['mask_address']
            full_img_path = os.path.join(self.root , img_path)
            mask = torchvision.io.read_image(full_img_path) 
            mask_lists.append(mask) 
            t_vec , r_mat = self.process_pose(item['cam_R_m2c'] , item['cam_t_m2c'])
            cam_r_m2c_list.append(r_mat)
            cam_t_m2c_list.append(t_vec) 
            obj_id = item['object_id'] 
            object_id_list.append(obj_id) 

        return mask_lists , cam_r_m2c_list , cam_t_m2c_list , object_id_list 

    def __getitem__(self, idx : int) :
        sample = self.all_samples[idx] 
        rgb_image = self._read_and_normalize_rgb_image(sample['rgb_image_adderss']) 
        depth_map_image = self._read_and_standardized_depth_img(sample['depth_map_image_address'] , sample['depth_scale'])
        mask_lists , cam_r_m2c_list , cam_t_m2c_list , object_id_list  = self._read_masks_and_other_meta_data(sample['laebels'])