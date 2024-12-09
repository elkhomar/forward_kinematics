import torch
import kinetix_scenegraph as ks
from kinetix_body_models.hg_body_model import HumanGen
import numpy as np
from kinetix_scenegraph.utils import rotation_conversions
import fire
from pathlib import Path
import json

def load_motion_data(npz_path: str):
    """Load motion data from NPZ file."""
    data = np.load(npz_path)
    
    # Get betas, poses and translations
    betas = torch.from_numpy(data['betas']).float()  # (1, 10)
    poses = torch.from_numpy(data['poses_resilient-glade_pred_world']).float()  # (1, 478, 102, 3)
    trans = torch.from_numpy(data['trans_resilient-glade_pred_world']).float()  # (1, 478, 3)

    return data, betas, poses, trans

def forward_kinematics(model_path: str = "/home/omar/Downloads/HumanGen.pkl", npz_path: str = "/home/omar/Kinetix/repo/kinetix-npz-to-bodymodel/data/KD1_X0032_Locomotion_Walking_6_Romain_30fps_ML_Romain_RT_Romain_Jonathan_1.npz"):
    """
    Perform forward kinematics using HumanGen model.
    
    Args:
        model_path: Path to HumanGen model file
        npz_path: Path to motion data NPZ file
    """
    # Load motion data
    data, betas, poses, trans = load_motion_data(npz_path)
    num_frames = poses.shape[1]

    # Initialize model
    scene = ks.Scene()
    scene.batch_size = num_frames
    model = HumanGen(model_path, parent=scene)
    
    
    # Reshape poses for the model
    # Original shape: (1, 478, 102, 3)
    
    model.betas = betas[0]
    model.skeleton.bone_list[0].set_global_positions(trans[0])

    # Only the body joints are animated (26) the npz contains a skeleton with 102 joints and the body model we instantiated has 64 joints
    # We need to map the 102 joints to the 64 joints
    body_joints_102 = [0, 1, 2, 3, 4, 5, 44, 45, 46, 47, 67, 68, 69, 70, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
    body_joints_64 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 30, 31, 32, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    rotation_matrices = rotation_conversions.axis_angle_to_matrix(poses[0])
    body_bones = [model.skeleton.bone_list[i] for i in body_joints_64]
    for idx, bone in enumerate(body_bones):
        bone.set_local_rotations_matrices(rotation_matrices[:, body_joints_102[idx]])
    
    initial_rotation = model.skeleton.bone_list[0].get_local_rotations_matrices()[0]
    #model.skeleton.bone_list[0].apply_global_rotations_quaternion(rotation_conversions.matrix_to_quaternion(torch.linalg.inv(initial_rotation)))
    offset = torch.tensor([[-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0, 0.0, -1.0]])
    model.skeleton.bone_list[0].set_global_positions(trans[0]@offset.T)
    model.skeleton.bone_list[0].apply_global_rotations_quaternion(rotation_conversions.matrix_to_quaternion(offset))
    
    kp_3d = {bone.name: bone.get_global_positions() for bone in body_bones}
    global_rotations = {bone.name: bone.get_global_rotations_matrices()[:] for bone in body_bones}

    for idx, bone in enumerate(body_bones):
        bone.axis = ks.primitives.Axes(name=f"{bone.name}_axis", parent=scene)
        bone.axis.set_global_rotations_matrix(bone.get_global_rotations_matrices())
        bone.axis.set_global_positions(bone.get_global_positions())

    hierarchy = {bone.name: bone.parent.name for bone in body_bones}

    export_dict = {"3d_keypoints": kp_3d,
                   "global_rotations": global_rotations,
                   "hierarchy": hierarchy}
    
    # Create coordinate axis visualization spheres
    x_sphere = ks.primitives.Sphere(name = "x_axis", parent=scene)
    y_sphere = ks.primitives.Sphere(name = "y_axis", parent=scene)
    z_sphere = ks.primitives.Sphere(name = "z_axis", parent=scene)

    x_sphere.set_global_positions(torch.tensor([[1.0, 0.0, 0.0]]))
    y_sphere.set_global_positions(torch.tensor([[0.0, 1.0, 0.0]]))
    z_sphere.set_global_positions(torch.tensor([[0.0, 0.0, 1.0]]))

    x_sphere.set_global_scales(torch.tensor([[0.05, 0.05, 0.05]]))
    y_sphere.set_global_scales(torch.tensor([[0.05, 0.05, 0.05]]))
    z_sphere.set_global_scales(torch.tensor([[0.05, 0.05, 0.05]]))
 
    return export_dict

if __name__ == "__main__":
    fire.Fire(forward_kinematics)
