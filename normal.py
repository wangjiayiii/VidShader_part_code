"""
Render NPHM-normal maps and convert them into video sequences.

This script:
1. Loads per-frame mesh + NPHM transforms
2. Loads FLAME camera parameters
3. Converts OpenCV camera → PyTorch3D camera
4. Renders per-frame normal maps
5. Exports PNGs and packs them into MP4 videos
"""

import os
import re
import argparse
import numpy as np
import torch
import cv2

from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    PerspectiveCameras,
)

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#                   Utility Functions
# ============================================================

def load_ply_file(obj_dir, device="cuda"):
    """Load mesh.ply vertices & faces."""
    import trimesh
    mesh = trimesh.load(os.path.join(obj_dir, 'mesh.ply'))
    verts = torch.tensor(mesh.vertices.copy(), dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces.copy(), dtype=torch.int64, device=device)
    return verts, faces


def load_flame_camera_params(camera_path, device="cuda"):
    """Load FLAME initial camera parameters from OpenCV .npz file."""
    camera = np.load(camera_path)
    R = torch.tensor(camera["R"], device=device)
    T = torch.tensor(camera["t"], device=device)
    K = torch.tensor(camera["K"], device=device)
    return R, T, K


def load_nphm_transforms(obj_dir, device="cuda"):
    """Load NPHM-optimized rotation, scale, translation."""
    rot = torch.tensor(np.load(os.path.join(obj_dir, "rot.npy")),
                       dtype=torch.float32, device=device)
    scale = torch.tensor(np.load(os.path.join(obj_dir, "scale.npy")),
                         dtype=torch.float32, device=device)
    trans = torch.tensor(np.load(os.path.join(obj_dir, "trans.npy")),
                         dtype=torch.float32, device=device)
    return rot, scale, trans


def apply_nphm_transforms(verts, rot, scale, trans):
    """Apply NPHM transforms to mesh vertices."""
    rot, scale, trans = rot[0], scale[0], trans[0]

    verts = verts * scale
    verts = torch.mm(verts, rot.transpose(0, 1))
    verts = verts + trans
    return verts


# ============================================================
#                        Rendering
# ============================================================

def render_normal_map(obj_dir, sequence_name, id_folder, cam_folder,
                      frame_id, device="cuda"):
    """
    Render 3D mesh into a normal map using PyTorch3D according to
    FLAME camera parameters & NPHM transforms.
    """
    try:
        # ----------------------------- #
        #  Load FLAME camera parameters
        # ----------------------------- #
        camera_path = (
            f"/mnt/hd1/nerfsemble_7_11/{sequence_name}/{id_folder}/{cam_folder}/"
            f"preprocess/metrical_tracker/{sequence_name}_{id_folder}_{cam_folder}/"
            f"checkpoint/{frame_id:05d}_cam_params_opencv.npz"
        )

        R, T, K = load_flame_camera_params(camera_path)

        # ----------------------------- #
        #  Load + transform mesh
        # ----------------------------- #
        verts, faces = load_ply_file(obj_dir)
        verts *= 0.25  # FLAME default scale

        rot, scale, trans = load_nphm_transforms(obj_dir, device)
        verts = apply_nphm_transforms(verts, rot, scale, trans)

        # ----------------------------- #
        #  Convert OpenCV → PyTorch3D
        # ----------------------------- #
        conversion_matrix = torch.tensor(
            [[-1, 0, 0],
             [0, -1, 0],
             [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )

        R_p3d = torch.matmul(conversion_matrix, R).transpose(1, 2)
        T_p3d = T.clone()
        T_p3d[:, :2] *= -1  # Flip x, y translation

        image_size = torch.tensor([[1024, 1024]])
        focal_length = K[0].diag()[:2].unsqueeze(0)
        principal_point = K[:, :2, 2]

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R_p3d,
            T=T_p3d,
            in_ndc=False,
            image_size=image_size,
            device=device,
        )

        # ----------------------------- #
        #  Render mesh
        # ----------------------------- #
        mesh = Meshes(verts=[verts], faces=[faces])

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=(1024, 1024),
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_backfaces=True,
                perspective_correct=True,
            ),
        )

        fragments = rasterizer(mesh)

        # Compute face normals
        face_verts = mesh.verts_padded()[0][mesh.faces_padded()[0]]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        face_normals = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0), dim=1)

        pix_to_face = fragments.pix_to_face[0, ..., 0]
        H, W = pix_to_face.shape
        normal_img = torch.zeros((H, W, 3), device=device)

        valid = pix_to_face >= 0
        normal_img[valid] = (face_normals[pix_to_face[valid]] + 1.0) / 2.0

        return (255 * normal_img.detach().cpu().numpy()).astype(np.uint8)

    except Exception as e:
        print(f"[ERROR] Rendering failed at frame {frame_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
#                    Sequence-Level Processing
# ============================================================

def create_video_from_renders(sequence_name, id_folder, cam_folder,
                              input_base_path, output_base_path):
    """
    Render all PNG frames for a given ID/cam, then pack them into a video.
    """
    sequence_path = os.path.join(input_base_path, id_folder, cam_folder)

    # Camera checkpoint directory
    camera_ckpt_dir = (
        f"/mnt/hd1/nerfsemble_7_11/{sequence_name}/{id_folder}/{cam_folder}/"
        f"preprocess/metrical_tracker/{sequence_name}_{id_folder}_{cam_folder}/checkpoint"
    )

    # Temporary output directory
    temp_dir = os.path.join(output_base_path, sequence_name, id_folder, "tmp", cam_folder)
    os.makedirs(temp_dir, exist_ok=True)

    # ----------------------------- #
    #  Collect valid frame IDs
    # ----------------------------- #
    valid_frame_ids = set()
    if os.path.exists(camera_ckpt_dir):
        for fname in os.listdir(camera_ckpt_dir):
            match = re.match(r"^(\d{5})_cam_params_opencv\.npz$", fname)
            if match:
                valid_frame_ids.add(int(match.group(1)))

    if not valid_frame_ids:
        print(f"[WARN] No camera params found in: {camera_ckpt_dir}")
        return

    rendered_images = []

    # ----------------------------- #
    #  Render all frames
    # ----------------------------- #
    for frame_id in tqdm(sorted(valid_frame_ids),
                         desc=f"Rendering {id_folder}/{cam_folder}"):

        frame_dir = os.path.join(sequence_path, "preprocess", f"{frame_id:05d}")
        mesh_file = os.path.join(frame_dir, "mesh.ply")

        if not os.path.exists(mesh_file):
            continue

        img_np = render_normal_map(frame_dir, sequence_name, id_folder, cam_folder, frame_id)
        if img_np is None:
            continue

        out_img_path = os.path.join(temp_dir, f"{frame_id:05d}.png")
        cv2.imwrite(out_img_path, img_np)
        rendered_images.append(out_img_path)

    if not rendered_images:
        print(f"[WARN] No frames rendered for: {id_folder}/{cam_folder}")
        return

    # ----------------------------- #
    #  Write video
    # ----------------------------- #
    video_out_dir = os.path.join(output_base_path, sequence_name, id_folder, "train_dwpose")
    os.makedirs(video_out_dir, exist_ok=True)

    video_path = os.path.join(video_out_dir, f"{sequence_name}_{cam_folder}.mp4")

    sample = cv2.imread(rendered_images[0])
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (w, h),
    )

    for img_path in sorted(rendered_images):
        img = cv2.imread(img_path)
        if img is not None:
            writer.write(img)

    writer.release()

    print(f"[INFO] Video saved at: {video_path}")
    print(f"[INFO] PNG frames saved at: {temp_dir}")


# ============================================================
#                        Main Entrypoint
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        default="/mnt/hd1/nerfsemble_7_11/pretrained_mononphm_original/stage2",
    )
    parser.add_argument(
        "--output_folder",
        default="/mnt/hd1/nerfsemble_7_11/result",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sequence_name = "sequence_EXP-3-cheeks+nose_part-1"
    id_folder = "112"
    cam_folder = "cam_222200036"

    args.input_folder = (
        f"/mnt/hd1/nerfsemble_7_11/pretrained_mononphm_original/stage2/{sequence_name}"
    )
    args.output_folder = "/mnt/hd1/nerfsemble_7_11/result"

    create_video_from_renders(
        sequence_name,
        id_folder,
        cam_folder,
        args.input_folder,
        args.output_folder,
    )
