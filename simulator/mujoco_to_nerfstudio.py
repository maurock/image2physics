import mujoco
from PIL import Image
import os
from utils_dir.constants import Paths, Resolution
from utils_dir import utils_mujoco
import argparse
import random
import string


def parse_arguments():
    args = argparse.ArgumentParser(description="Generate NeRFStudio-compatible scenes from MuJoCo")
    args.add_argument('--scene_name', type=str, default='', help="Name of the scene, default is a random string.")
    args.add_argument('--num_cameras', type=int, default=50, help="Number of cameras to generate.")
    args.add_argument('--radius', type=int, default=20, help="Radius of camera placement around the target.")
    return args.parse_args()


def generate_scene_name(scene_name:str) -> str:
    if len(scene_name) > 0:
        return scene_name
    else:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    
   
def main():

    args = parse_arguments()

    scene_name = generate_scene_name(args.scene_name)

    # Create scene and image directories
    scene_dir = os.path.join(Paths.SCENES.value, scene_name)
    image_dir = os.path.join(scene_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)

    h, w = Resolution.SQUARE.value  

    # Load model and data  
    model = utils_mujoco.load_model_with_cameras('ycb_scene.xml', num_cameras=args.num_cameras, radius=args.radius)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=w, height=h) 

    # Create scene data (transforms.json)
    intrinsics = utils_mujoco.extract_camera_intrinsics(model=model, camera_id=0, width=w, height=h)
    frames = utils_mujoco.render_images(renderer, data, args.num_cameras, scene_dir)
    transforms = {**intrinsics, "frames": frames}   

    # Save scene data
    utils_mujoco.save_transforms_json(transforms, scene_name)
    renderer.close()
  

if '__main__'==__name__:
    main()

