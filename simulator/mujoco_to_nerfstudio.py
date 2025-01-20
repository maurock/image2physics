import mujoco
from PIL import Image
import os
from utils_dir.constants import Paths, Resolution
from utils_dir import utils_mujoco
import argparse
import random
import string
from custom_types import ConfigFile
import yaml
from dm_control import mjcf
from typing import Sequence
from dataclasses import dataclass

@dataclass
class YCB_Paths:
    mesh_paths: Sequence[str]
    texture_paths: Sequence[str]


def parse_arguments() -> argparse.Namespace:
    args = argparse.ArgumentParser(description="Generate NeRFStudio-compatible scenes from MuJoCo")
    args.add_argument('--scene_name', type=str, default='', help="Name of the scene, default is a random string.")
    args.add_argument('--num_cameras', type=int, default=50, help="Number of cameras to generate.")
    args.add_argument('--radius', type=int, default=20, help="Radius of camera placement around the target.")
    args.add_argument('--dataset_name', type=str, default='default', help="Type of dataset to load: ['default', 'ycb']")    
        
    return args.parse_args()


def generate_scene_name(scene_name:str) -> str:
    if len(scene_name) > 0:
        return scene_name
    else:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    

def get_base_config(**kwargs) -> ConfigFile:
    config_data = """
    dataset:
        name: ycb
        ycb_object_id: 9
        cameras: 30
    """

    return ConfigFile(yaml.safe_load(config_data))


def get_ycb_paths() -> Sequence[YCB_Paths]:
    """Returns the paths to the YCB objects."""

    ycb_dir = os.path.join(Paths.ASSETS.value, 'ycb')
    
    
    mesh_paths = [os.path.join(ycb_dir, obj, 'google_64k', 'textured.obj') for obj in os.listdir(ycb_dir)]
    texture_paths = [os.path.join(ycb_dir, obj, 'google_64k', 'texture_map.png') for obj in os.listdir(ycb_dir)]
    ycb_paths = YCB_Paths(mesh_paths=mesh_paths, texture_paths=texture_paths)
    
    return ycb_paths


def make_model(args: argparse.Namespace, config_data: ConfigFile) -> mujoco.MjModel:
    
    if args.dataset_name == 'default':
        model_name = 'ycb_scene.xml'
        
        model_xml = utils_mujoco.load_model_with_cameras(model_name=model_name, num_cameras=args.num_cameras, radius=args.radius)
        mj_model = mujoco.MjModel.from_xml_path(model_xml)

        return mj_model
    
    elif args.dataset_name == 'ycb':
        ycb_paths = get_ycb_paths()

        model_name = 'mujocosim.xml'
        
        obj_id = config_data['dataset']['ycb_object_id']
       
        with open(ycb_paths.mesh_paths[obj_id], 'rb') as f:
            mesh_file = mjcf.Asset(f.read(), '.obj')
        with open(ycb_paths.texture_paths[obj_id], 'rb') as f:
            texture_file = mjcf.Asset(f.read(), '.png')
        
        name = 'mujocosim.xml'
        model_dir = os.path.join(Paths.MJ_MODELS.value, name)
        mjcf_root = mjcf.from_path(model_dir)
        
        print(mjcf_root.to_xml_string())

        mjcf_root.asset.add(
            element_name='mesh', name=name, file=mesh_file
        )
        mjcf_root.asset.add(
            element_name='texture', name=name, file=texture_file, type='2d'
        )
        mjcf_root.asset.add(
            element_name='material', name=name, texture=name
        )
        body = mjcf_root.worldbody.add('body', name=name)
        body.add('geom', type='mesh', mesh=name, material=name)
        
        print(mjcf_root.to_xml_string())

        mjcf_root = utils_mujoco.add_cameras_to_mjcf(
            mjcf_root,
            num_cameras=config_data['dataset']['cameras'],
            radius=args.radius
        )

        mj_model = mjcf.Physics.from_mjcf_model(mjcf_root).model.ptr

    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    return mj_model


def main(**kwargs):
    args = parse_arguments()
    
    config_data = get_base_config()

    if kwargs and kwargs.get('ycb_object_id'):
        config_data['dataset']['ycb_object_id'] = kwargs["ycb_object_id"]

    # scene_name = generate_scene_name(args.scene_name)
    ycb_dir = os.path.join(Paths.ASSETS.value, 'ycb')
    scene_name = [obj for obj in os.listdir(ycb_dir)][config_data['dataset']['ycb_object_id']]
    
    w, h = 512, 512

    # Create scene and image directories
    scene_dir = os.path.join(Paths.SCENES.value, scene_name)
    image_dir = os.path.join(scene_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    # Load model and data
    model = make_model(args, config_data)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Create scene data (transforms.json)
    intrinsics = utils_mujoco.extract_camera_intrinsics(model=model, camera_id=0, width=w, height=h)
    frames = utils_mujoco.render_images(model, data, args.num_cameras, scene_dir, width=w, height=h)
    transforms = {**intrinsics, "frames": frames}   

    # Save scene data
    utils_mujoco.save_transforms_json(transforms, scene_name)
  

if '__main__'==__name__:
    for i in range(0,64):
        try:
            main(ycb_object_id=i)
        except:
            continue
