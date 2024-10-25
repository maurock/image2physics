import os
import subprocess
import shutil
from utils import get_last_file_in_folder, get_scenes_list, project_3d_to_2d, load_depths, parse_dataparser_transforms_json
from arguments import get_args
import torch
from nerfstudio.scripts.exporter import ExportPointCloud
from pathlib import Path
from segmentation import Segmentation
from utils_dir.constants import Paths
import numpy as np
from dataclasses import dataclass
from PIL import Image
import json
from utils import load_ns_point_cloud
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
from rich.console import Console
from plotly import graph_objs as go

@dataclass
class Camera:
    idx: int = None
    img_path: str = None
    img: np.ndarray = None
    w2c: np.ndarray = None
    K: np.ndarray = None
    masks: np.ndarray = None
    labels: list = None
    depth: np.ndarray = None

@dataclass
class Struct3D:
    label: str = None
    mesh: o3d.geometry.TriangleMesh = None
    point_cloud: np.ndarray = None
    colors: np.ndarray = None

console = Console()

def move_files_to_folder(source_dir, target_dir):
    for file in os.listdir(source_dir):
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))


class CameraProcessor:
    """Processes and loads camera data from a scene."""
    
    def __init__(self, scene_dir: str, segmentation_model: Segmentation = None):
        self.scene_dir = scene_dir
        self.segmentation = segmentation_model
    
    def create_cameras(self, text_prompt: str):
        """Create cameras for the scene."""
        console.print('Creating cameras..', style='bold green')

        images_dir = os.path.join(self.scene_dir, 'images')
        transforms_path = os.path.join(self.scene_dir, 'transforms.json') 
        depth_dir = os.path.join(self.scene_dir, 'ns', 'renders', 'depth')
        depths = load_depths(depth_dir, Ks=None)
        
        cameras = []
        image_files = os.listdir(images_dir)
        # sort numerically
        image_files = sorted(image_files, key=lambda x: int(x[3:x.index('.png')]))
        for idx, image_file in enumerate(tqdm(image_files)):
            image_path = os.path.join(images_dir, image_file)
            image = np.array(Image.open(image_path))
            if self.segmentation is not None:
                masks, results = self.segmentation.predict_masks(image_path, text_prompt)
                labels = results[0]['labels']
                masks, labels = self._filter_empty_labels(masks, labels)
            else:
                masks, labels = None, None
            w2c = np.linalg.inv(self.get_c2w_from_transforms(transforms_path, idx))
            K = self.get_K_from_transforms(transforms_path)
            depth = depths[idx]
            cameras.append(Camera(idx, image_path, image, w2c, K, masks, labels, depth))
        return cameras
    
    @staticmethod
    def get_c2w_from_transforms(transforms_path, idx) -> np.ndarray:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        return np.array(transforms['frames'][idx]['transform_matrix'])
    
    @staticmethod
    def get_K_from_transforms(transforms_path) -> np.ndarray:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        return np.array([
            [transforms["fl_x"], 0, transforms["cx"]],
            [0, transforms["fl_y"], transforms["cy"]],
            [0, 0, 1]
        ])

    def _filter_empty_labels(self, masks, labels):
        """Filter out empty masks and labels."""
        new_masks, new_labels = [], []
        for mask, label in zip(masks, labels):
            if label: # Only keep non-empty labels
                new_masks.append(mask)
                new_labels.append(label)
        return new_masks, new_labels


class PointCloudFilter:
    """Handles the filtering of point clouds based on masks, depth, and occlusion."""
    
    def __init__(self, occlusion_threshold: float=0.):
        self.occ_thr = occlusion_threshold
    
    def filter_by_bounds(self, points2D, pcd, mask, dists):
        """Filter the point cloud by the bounds of the mask."""
        x_coords = points2D[:, 0].astype(int)
        y_coords = points2D[:, 1].astype(int)
        in_bounds = (x_coords >= 0) & (x_coords < mask.shape[1]) & (y_coords >= 0) & (y_coords < mask.shape[0])
        valid_points2D, valid_points3D = points2D[in_bounds], np.asarray(pcd.points)[in_bounds]
        valid_normals = np.asarray(pcd.normals)[in_bounds]
        valid_dists = dists[in_bounds]
       	valid_colors = np.asarray(pcd.colors)[in_bounds]
        return valid_points2D, valid_points3D, valid_dists, valid_normals, valid_colors 
    
    def filter_by_occlusion(self, depth, points2D, points3D, dists, normals, colors):
        is_occluded = dists > depth[points2D[:, 1].astype(int), points2D[:, 0].astype(int)] + self.occ_thr
        return points2D[~is_occluded], points3D[~is_occluded], normals[~is_occluded], colors[~is_occluded]

    def filter_by_mask(self, points2D, points3D, mask, normals, colors):
        x_coords = points2D[:, 0].astype(int)
        y_coords = points2D[:, 1].astype(int)
        mask_values = mask[y_coords, x_coords]
        return points2D[mask_values == 1], points3D[mask_values == 1], normals[mask_values == 1], colors[mask_values == 1]


class MeshExtractor:
    """Extracts individual meshes per object from segmented point clouds."""
    
    def __init__(self, pcd: o3d.geometry.PointCloud, cameras: list, occ_thr, dt_file: str):
        self.pcd = pcd
        self.cameras = cameras
        _, self.scale = parse_dataparser_transforms_json(dt_file)
        self.occ_thr = occ_thr * self.scale
        self.point_cloud_filter = PointCloudFilter(self.occ_thr)

    def extract(self, nb_neighbors=20, std_ratio=10.):
        """Extracts 3D meshes for each label by processing the camera images and masks.

        Returns:
            results (list): A list of `Struct3D` objects, where each object contains the label,
                            corresponding 3D mesh, and point cloud."""
        console.print('Extracting meshes..', style='bold green')
        label_dict = defaultdict(lambda: defaultdict(list))
        for camera in self.cameras:
            for idx, mask in enumerate(camera.masks):
                label = camera.labels[idx]
                projected_2D, dists = project_3d_to_2d(np.asarray(self.pcd.points), camera.w2c, camera.K, return_dists=True)
                pts2D, pts3D, dists, normals, colors = self.point_cloud_filter.filter_by_bounds(projected_2D, self.pcd, mask, dists)
                pts2D, pts3D, normals, colors = self.point_cloud_filter.filter_by_occlusion(camera.depth, pts2D, pts3D, dists, normals, colors)
                pts2D, pts3D, normals, colors = self.point_cloud_filter.filter_by_mask(pts2D, pts3D, mask, normals, colors)
                label_dict[label]['pointcloud'].extend(pts3D)
                label_dict[label]['normals'].extend(normals)
                label_dict[label]['colors'].extend(colors)
        
        # Unique filtering and mesh creation
        results = []
        for label in label_dict.keys():
            points, indices = np.unique(np.array(label_dict[label]['pointcloud']), return_index=True, axis=0)
            normals = np.array(label_dict[label]['normals'])[indices]
            colors = np.array(label_dict[label]['colors'])[indices]
            mesh = self.create_mesh(points, normals, colors, nb_neighbors, std_ratio)
            results.append(Struct3D(label, mesh, points, colors))

        return results

    @staticmethod
    def create_mesh(point_cloud, normals, colors, nb_neighbors, std_ratio):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh
    
    def _plot_point_cloud(self, results, label):
        """Plot the point cloud in 3D."""
        for result in results:
            if result.label == label:
                point_cloud = result.point_cloud

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd])
    
    def _plot_points_on_image(self, points2D, H, W):
        """
        Debug plot given pixel indices (points2D) on an empty HxW grayscale image.
        """
        image = np.zeros((H, W), dtype=np.uint8)
        x_coords = points2D[:, 0].astype(int)
        y_coords = points2D[:, 1].astype(int)
        image[y_coords, x_coords] = 255
        plt.imshow(image, cmap='gray')
        plt.show()        
        return image
    

if __name__ == '__main__':
    
    args = get_args()

    console = Console()   # pretty logging
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = DEVICE

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)


    for scene in scenes: 
        base_dir = os.path.join(scenes_dir, scene, 'ns')

        # Calling ns-train
        result = subprocess.run([
            'ns-train', 'nerfacto', 
            '--data', os.path.join(scenes_dir, scene),
            '--output_dir', base_dir,
            '--vis', args.vis_mode,
            '--project_name', args.project_name,
            '--experiment_name', scene,
            '--max_num_iterations', str(args.training_iters),
            '--pipeline.model.background-color', 'white',   # random
            '--pipeline.model.camera-optimizer.mode', 'off',
            '--pipeline.model.proposal-initial-sampler', 'uniform',
            '--pipeline.model.near-plane', str(args.near_plane),
            '--pipeline.model.far-plane', str(args.far_plane),
            '--steps-per-eval-image', '10000',
        ])

        ns_dir = get_last_file_in_folder(os.path.join(base_dir, '%s/nerfacto' % scene))

        # Copying dataparser_transforms (contains scale)
        result = subprocess.run([
            'scp', '-r', 
            os.path.join(ns_dir, 'dataparser_transforms.json'), 
            os.path.join(base_dir, 'dataparser_transforms.json')
        ])

        half_bbox_size = args.bbox_size / 2

        print('Exporting pointcloud..')
        pcd_exporter = ExportPointCloud(
            load_config=Path(os.path.join(ns_dir, 'config.yml')), 
            output_dir=Path(base_dir), 
            num_points=args.num_points, 
            remove_outliers=True, 
            normal_method='open3d', 
            obb_center=(0.0, 0.0, 0.0),
            obb_rotation=(0.0, 0.0, 0.0),
            obb_scale=(2., 2., 2.)
        )
        pcd_exporter.main()

        # Calling ns-render 
        result = subprocess.run([
            'ns-render', 'dataset',
            '--load-config', os.path.join(ns_dir, 'config.yml'),
            '--output-path', os.path.join(base_dir, 'renders'),
            '--rendered-output-names', 'raw-depth',
            '--split', 'train+test',
        ])

        # Collect all depths in one folder
        os.makedirs(os.path.join(base_dir, 'renders', 'depth'), exist_ok=True)
        move_files_to_folder(os.path.join(base_dir, 'renders', 'test', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))
        move_files_to_folder(os.path.join(base_dir, 'renders', 'train', 'raw-depth'), os.path.join(base_dir, 'renders', 'depth'))

        # Set up directories and file paths
        pcd_file = os.path.join(base_dir, 'point_cloud.ply')
        dt_file = os.path.join(base_dir, 'dataparser_transforms.json')

        segmentation = Segmentation(
            grounding_model_id='IDEA-Research/grounding-dino-tiny',
            sam2_checkpoint=os.path.join(Paths.MODELS.value, "grounded-sam2", "sam2_hiera_large.pt"),
            sam2_config="sam2_hiera_l.yaml"
        )

        # Load pointcloud
        console.print('Loading point cloud..', style='bold green')
        full_pcd = load_ns_point_cloud(pcd_file, dt_file)
        
        camera_processor = CameraProcessor(
            os.path.join(scenes_dir, scene),
            segmentation_model=segmentation
        )
        cameras = camera_processor.create_cameras(args.text_prompt)

        # Mesh extraction
        mesh_extractor = MeshExtractor(full_pcd, cameras, args.occ_thr, dt_file)
        results = mesh_extractor.extract()

        # Save meshes
        for struct3D in results:
            os.makedirs(os.path.join(scenes_dir, scene, 'meshes'), exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(scenes_dir, scene, 'meshes', f'{struct3D.label}.ply'), struct3D.mesh)
