import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    BlendParams, PerspectiveCameras, SoftPhongShader, PointLights
)
from pytorch3d.renderer.mesh import TexturesVertex


# Force CUDA usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ply_file(obj_dir, device="cuda"):
    """
    Load a mesh in PLY format and return vertices and faces.
    """
    import trimesh
    mesh = trimesh.load(os.path.join(obj_dir, "mesh.ply"))
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.float32, device=device)
    return verts, faces


def load_camera_params(camera_path, device="cuda"):
    """
    Load camera parameters from .npz file.
    """
    cam = np.load(camera_path)
    R = torch.tensor(cam["R"], device=device)
    T = torch.tensor(cam["t"], device=device)
    K = torch.tensor(cam["K"], device=device)
    return R, T, K


def convert_opencv_to_pytorch3d(R, T, device="cuda"):
    """
    Convert OpenCV camera coordinates to PyTorch3D coordinates.
    """
    conversion_matrix = torch.tensor([
        [-1, 0, 0],
        [0, -1, 0],
        [0,  0, 1]
    ], dtype=torch.float32, device=device)

    R_new = (conversion_matrix @ R).transpose(1, 2)
    T_new = T.clone()
    T_new[:, :2] *= -1

    return R_new, T_new


def render_frame(obj_dir, sequence_name, id_folder, cam_folder, frame_id, device="cuda"):
    """
    Render one frame using PyTorch3D, given mesh and camera parameters.
    """
    cam_path = (
        f"/mnt/hd/cropped_nersemble_other/batch1/{sequence_name}/"
        f"{id_folder}/{cam_folder}/metrical_tracker/{sequence_name}_{id_folder}/"
        f"checkpoint/{frame_id:05d}_cam_params_opencv.npz"
    )

    # Load mesh and camera
    verts, faces = load_ply_file(obj_dir)
    verts *= 0.25
    R, T, K = load_camera_params(cam_path)
    R, T = convert_opencv_to_pytorch3d(R, T)

    # Create mesh textures (white)
    vertex_colors = torch.ones_like(verts).unsqueeze(0).to(device)
    textures = TexturesVertex(verts_features=vertex_colors)

    mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(device)

    # Camera setup
    image_size = torch.tensor([[1024, 1024]])
    focal = K[0].diag()[:2].unsqueeze(0)
    pp = K[:, :2, 2]

    cameras = PerspectiveCameras(
        focal_length=focal,
        principal_point=pp,
        R=R, T=T,
        image_size=image_size,
        in_ndc=False,
        device=device
    )

    # Renderer
    raster_settings = RasterizationSettings(
        image_size=(1024, 1024),
        blur_radius=0.0,
        faces_per_pixel=100
    )

    lights = PointLights(location=[[0.0, 0.0, 3.0]], device=device)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(cameras=cameras, lights=lights, device=device)
    )

    image = renderer(mesh)[0, ..., :3]
    img_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    return img_np


def create_video_from_renders(sequence_name, id_folder, cam_folder, input_base, output_base):
    """
    Render all frames for a given (sequence, ID, camera) into a video file.
    """
    seq_path = os.path.join(input_base, id_folder, cam_folder, "preprocess")
    frame_dirs = sorted([d for d in os.listdir(seq_path) if d.isdigit()])

    if not frame_dirs:
        print(f"No frame directories found in: {seq_path}")
        return

    tmp_dir = os.path.join(output_base, id_folder, "tmp", sequence_name)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    rendered = []

    for fdir in tqdm(frame_dirs, desc=f"Rendering {id_folder}/{cam_folder}"):
        fpath = os.path.join(seq_path, fdir)
        if not os.path.exists(os.path.join(fpath, "mesh.ply")):
            continue

        frame_id = int(fdir)

        try:
            img = render_frame(fpath, sequence_name, id_folder, cam_folder, frame_id)
            out_path = os.path.join(tmp_dir, f"{frame_id:05d}.png")
            cv2.imwrite(out_path, img)
            rendered.append(out_path)
        except Exception as e:
            print(f"Error rendering frame {frame_id}: {e}")

    if not rendered:
        print(f"No frames rendered for {id_folder}/{cam_folder}")
        return

    # Build video
    output_dir = os.path.join(output_base, id_folder, "train_dwpose")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    video_path = os.path.join(output_dir, f"{sequence_name}_{cam_folder}.mp4")

    first = cv2.imread(rendered[0])
    h, w = first.shape[:2]
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

    for img_path in sorted(rendered):
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)

    out.release()

    print(f"Video saved to: {video_path}")
    print(f"Rendered PNGs stored in: {tmp_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sequence_name = "sequence_EXP-3-cheeks+nose_part-1"
    id_folder = "112"
    cam_folder = "cam_220700191"

    create_video_from_renders(
        sequence_name,
        id_folder,
        cam_folder,
        args.input_folder,
        args.output_folder
    )
