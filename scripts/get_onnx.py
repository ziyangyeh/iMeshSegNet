import os
import sys

sys.path.append(os.getcwd())

import argparse
import time

import numpy as np
import torch
import torch.onnx
import vedo
from omegaconf import OmegaConf

from models import LitModule
from infer import get_metadata

torch.set_default_tensor_type("torch.FloatTensor")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_onnx(cfg_path: str, ckpt_path: str, mesh_file: str, filepath: str):
    cfg = OmegaConf.load(cfg_path)
    module = LitModule(cfg).load_from_checkpoint(ckpt_path)
    device = module.device
    model = module.model
    model.eval()

    mesh = vedo.Mesh(mesh_file)
    start_time = time.time()
    N = mesh.ncells
    points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:, 1:]
    cells = points[ids].reshape(N, 9).astype(dtype="float32")
    normals = vedo.vedo2trimesh(mesh).face_normals
    barycenters = mesh.cell_centers()

    mesh_d = mesh.clone()
    predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
    input_data = get_metadata(cfg, mesh, device)

    infer_start_time = time.time()
    with torch.no_grad():
        if cfg.model.name == "iMeshSegNet":
            input_shape = input_data["input"].shape
            KG_12_shape = input_data["KG_12"].shape
            KG_6_shape = input_data["KG_6"].shape
            input_sample_shape = (
                tuple(input_shape),
                tuple(KG_12_shape),
                tuple(KG_6_shape),
            )
        elif cfg.model.name == "MeshSegNet":
            input_shape = input_data["input"].shape
            A_S_shape = input_data["A_S"].shape
            A_L_shape = input_data["A_L"].shape
            input_sample_shape = (
                tuple(input_shape),
                tuple(A_S_shape),
                tuple(A_L_shape),
            )

    x = tuple(
        [torch.randn(i, requires_grad=True, device=device) for i in input_sample_shape]
    )

    if cfg.model.name == "iMeshSegNet":
        torch.onnx.export(
            model,
            args=x,  # model input (or a tuple for multiple inputs)
            f=filepath,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input", "kg_12", "kg_6"],  # the model's input names
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "points_num"},
                "a_s": {0: "batch", 1: "points_num", 2: "points_num"},
                "a_l": {0: "batch", 1: "points_num", 2: "points_num"},
                "output": {0: "batch", 1: "points_num"},
            },
        )
    elif cfg.model.name == "MeshSegNet":
        torch.onnx.export(
            model,
            args=x,  # model input (or a tuple for multiple inputs)
            f=filepath,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input", "a_s", "a_l"],  # the model's input names
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "points_num"},
                "a_s": {0: "batch", 1: "points_num", 2: "points_num"},
                "a_l": {0: "batch", 1: "points_num", 2: "points_num"},
                "output": {0: "batch", 1: "points_num"},
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To ONNX Script")
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        metavar="",
        help="configuration file",
        default="config/default.yaml",
    )
    parser.add_argument(
        "-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", required=True
    )
    parser.add_argument(
        "-mesh", "--mesh_file", type=str, metavar="", help="mesh file", required=True
    )
    parser.add_argument(
        "-out", "--onnx_file", type=str, metavar="", help="onnx file", required=True
    )

    args = parser.parse_args()

    get_onnx(args.config_file, args.ckpt_path, args.mesh_file, args.onnx_file)
