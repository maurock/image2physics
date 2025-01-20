import data.assets as assets
import os
from utils_dir.constants import Paths
import numpy as np
import mujoco
from typing import Dict
import json
from PIL import Image
from dm_control import mjcf


def load_model(model_name: str) -> mujoco.MjModel:
    """Loads the MuJoCo model from the specified XML file. Default path defined
    in the Paths enum (Paths.MJ_MODELS.value)."""
    model_path = os.path.join(Paths.MJ_MODELS.value, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    return mujoco.MjModel.from_xml_path(model_path)


def read_xml(model_name: str) -> str:
    """Reads the XML file and returns the contents as a string."""
    model_path = os.path.join(Paths.MJ_MODELS.value, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    with open(model_path, 'r') as f:
        return f.read()


def append_cameras_to_xml(xml_model: str, xml_cameras: str) -> str:
    """Appends the camera definitions to the XML string."""
    insert_point = xml_model.find('</worldbody>')
    if insert_point == -1:
        raise ValueError("Cannot find '</worldbody>' tag in the XML.")
    cameras_xml = ''.join(xml_cameras)
    new_xml = xml_model[:insert_point] + cameras_xml + xml_model[insert_point:]

    return new_xml


def add_cameras_to_mjcf(
    mjcf_root: mjcf.RootElement,
    num_cameras: int,
    radius: int
) -> mujoco.MjModel:
    sampled_points = fibonacci_hemisphere_samples(num_cameras, radius)

    for i in range(num_cameras):
        xyaxes = compute_xyaxes(sampled_points[i])
        mjcf_root.worldbody.add(
            'camera',
            name=f'cam{i:02}',
            pos=(sampled_points[i][0], sampled_points[i][1], sampled_points[i][2]),
            xyaxes=f'{" ".join(map(str, xyaxes))}',
            resolution='512 512'
        )
    
    return mjcf_root


def load_model_with_cameras(
    model_name: str,
    num_cameras: int,
    radius: int
) -> mujoco.MjModel:
    """Loads the MuJoCo model from the specified XML file and adds cameras to it."""
    xml = read_xml(model_name)
    
    # Add cameras to the model
    xml_cameras = generate_camera_xml(num_cameras, radius=radius)

    # Append the camera XML to the model
    xml = append_cameras_to_xml(xml, xml_cameras)

    # Save the new XML to a temporary file. This is a workaround to load the model
    # with the correct paths for the included assets.
    output_temp_path = os.path.join(Paths.MJ_MODELS.value, 'temp.xml')
    with open(output_temp_path, 'w') as f:
        f.write(xml)
    
    # Load the model from the temporary XML file
    model = mujoco.MjModel.from_xml_path(output_temp_path)

    # Remove the temporary file
    # os.remove(output_temp_path)

    return model 


def extract_camera_extrinsics(data: mujoco.MjData, camera_id: int) -> np.ndarray:
    """Calculates the camera-to-world transformation matrix, which nerfstudio requires."""
    R = data.cam_xmat[camera_id].reshape(3, 3)
    t = data.cam_xpos[camera_id]

    # Construct camera-to-world transformation matrix
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = R
    camera_to_world[:3, 3] = t
    
    return camera_to_world # shape (4, 4)


def extract_camera_intrinsics(
    model: mujoco.MjModel,
    camera_id: int,
    width: int,
    height: int
) -> Dict:
    """Extracts camera intrinsic parameters."""
    fovy = model.cam_fovy[camera_id]
    focal_length_y = (0.5 * height) / np.tan(0.5 * fovy * np.pi / 180)
    focal_length_x = focal_length_y * (width / height)

    intrinsics = {
        "camera_model": "OPENCV", 
        "fl_x": focal_length_x,
        "fl_y": focal_length_y,
        "cx": width / 2,
        "cy": height / 2,
        "w": width,
        "h": height,
        "k1": 0.0,  # Assuming no distortion
        "k2": 0.0,  
        "p1": 0.0,  # Assuming no tangential distortion
        "p2": 0.0
    }
    return intrinsics


def generate_camera_xml(num_cameras, radius, lookat=[0, 0, 0]) -> str:
  """Generate an XML string for all the camera by sampling their positions on a hemisphere."""
  x, y, z = fibonacci_hemisphere_samples(num_cameras, radius)

  xml = ''
  for i in range(num_cameras):
      xyaxes = compute_xyaxes([x[i], y[i], z[i]], lookat)
      xml += f"""
      <camera name="cam{i}" pos="{x[i]} {y[i]} {z[i]}" xyaxes="{' '.join(map(str, xyaxes))}"/>
      """    

  return xml


def save_transforms_json(transforms: Dict, scene_name: str) -> None:
    """Saves the transforms data to a JSON file."""
    transforms_path = os.path.join(Paths.SCENES.value, scene_name, 'transforms.json')
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    print(f"Transforms saved to '{transforms_path}'")


def compute_xyaxes(cam_position: list, cam_lookat: list=[0., 0., 0.], up_vector: list=[0, 0, -1]):
    """
    Compute the xyaxes parameter for a camera in MuJoCo.
    
    Parameters:
    cam_position: position of the camera (x, y, z). shape (3, )
    cam_lookat: point the camera is looking at (x, y, z), shape (3,)
    up_vector: the world "up" direction. Defaults to [0, 0, 1].
        
    Returns:
    xyaxes: The computed xyaxes parameter, where the first three elements are the X axis
        and the next three elements are the Y axis of the camera's frame. shape (6, )
    """
    cam_position = np.array(cam_position)
    cam_lookat = np.array(cam_lookat)
    up_vector = np.array(up_vector)
    
    # Compute the Z axis (camera look direction)
    z_axis = - cam_lookat + cam_position
    z_axis = z_axis / (np.linalg.norm(z_axis) + 0.000001)  # Normalize the vector
    
    # Compute the X axis (cross product of Z axis and up_vector)
    # We reverse the order of the cross product to make the X axis orthogonal to both Z and "up"
    x_axis = np.cross(z_axis, up_vector)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 0.000001) # Normalize the vector
    
    # Compute the Y axis (cross product of Z axis and X axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize the vector
    
    # Combine X and Y axes to form the xyaxes parameter
    xyaxes = np.concatenate((x_axis, y_axis))
    
    return xyaxes


def fibonacci_hemisphere_samples(num_points, radius=1):
    indices = np.arange(0, num_points, dtype=float) + 0.5

    phi = np.arccos(1 - indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    sampled_points = np.stack([x, y, z], axis=1)

    return sampled_points


def render_images(model, data, num_cameras, scene_dir, width=512, height=512):
    """Render images using the renderer and save them."""
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        frames = []
        for i in range(num_cameras):
            frame = {}
            camera_name = f'cam{i:02}'

            # Render image
            renderer.update_scene(data, camera=camera_name)
            image = renderer.render()
            image_path = os.path.join(scene_dir, 'images', f'{camera_name}.png')
            Image.fromarray(image).save(image_path)

            # Add camera frame data
            frame['file_path'] = image_path
            frame['transform_matrix'] = extract_camera_extrinsics(data, i).tolist()
            frames.append(frame)

    return frames