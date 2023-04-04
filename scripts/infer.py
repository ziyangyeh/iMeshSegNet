import os
import sys

sys.path.append(os.getcwd())

import argparse
import ast
import time

import numpy as np
import pytorch_lightning as pl
import torch
import vedo
from omegaconf import OmegaConf
from pygco import cut_from_graph
from scipy.spatial import distance_matrix

from models import LitModule
from utils import get_graph_feature


def centring(mesh: vedo.Mesh):
    mesh.points(pts=mesh.points() - mesh.center_of_mass())
    return mesh


def get_metadata(cfg: OmegaConf, mesh: vedo.Mesh, device="cuda"):
    mesh = centring(mesh)
    N = mesh.ncells
    points = vedo.vtk2numpy(mesh.polydata().GetPoints().GetData())
    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:, 1:]
    cells = points[ids].reshape(N, 9).astype(dtype="float32")
    normals = vedo.vedo2trimesh(mesh).face_normals
    normals.setflags(write=1)
    barycenters = mesh.cell_centers()

    # normalized data
    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    for i in range(3):
        cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
        cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
        cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

    X = np.column_stack((cells, barycenters, normals))
    X = X.transpose(1, 0)

    meta = dict()
    meta["input"] = torch.from_numpy(X).unsqueeze(0).to(device, dtype=torch.float)

    if cfg.model.name == "iMeshSegNet":
        print("Getting KG6 and KG12.")
        KG_6 = get_graph_feature(
            torch.from_numpy(X[9:12, :]).unsqueeze(0), k=6
        ).squeeze(0)
        KG_12 = get_graph_feature(
            torch.from_numpy(X[9:12, :]).unsqueeze(0), k=12
        ).squeeze(0)
        meta["KG_6"] = KG_6.unsqueeze(0).to(device, dtype=torch.float)
        meta["KG_12"] = KG_12.unsqueeze(0).to(device, dtype=torch.float)
    elif cfg.model.name == "MeshSegNet":
        print("Getting A_S and A_L.")
        X = X.transpose(1, 0)
        # computing A_S and A_L
        A_S = np.zeros([X.shape[0], X.shape[0]], dtype="float32")
        A_L = np.zeros([X.shape[0], X.shape[0]], dtype="float32")
        D = distance_matrix(X[:, 9:12], X[:, 9:12])
        A_S[D < 0.1] = 1.0
        A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        A_L[D < 0.2] = 1.0
        A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

        # numpy -> torch.tensor
        A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
        A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
        meta["A_S"] = torch.from_numpy(A_S).to(device, dtype=torch.float)
        meta["A_L"] = torch.from_numpy(A_L).to(device, dtype=torch.float)

    return meta


def infer(
    cfg_path: str, ckpt_path: str, mesh_file: str, refine: bool, device="cuda"
) -> vedo.Mesh:
    cfg = OmegaConf.load(cfg_path)
    if len(cfg.infer.devices) == 1 and cfg.infer.accelerator == "gpu":
        device = f"cuda:{cfg.infer.devices[0]}"
    elif len(cfg.infer.devices) > 1 and cfg.infer.accelerator == "gpu":
        device = "cuda:0"
    module = LitModule(cfg).load_from_checkpoint(ckpt_path)
    model = module.model.to(device)
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
            tensor_prob_output = model(
                input_data["input"], input_data["KG_12"], input_data["KG_6"]
            )
        elif cfg.model.name == "MeshSegNet":
            tensor_prob_output = model(
                input_data["input"], input_data["A_S"], input_data["A_L"]
            )
    print("Inference time: ", time.time() - infer_start_time)
    patch_prob_output = tensor_prob_output.cpu().numpy()

    for i_label in range(cfg.model.num_classes):
        predicted_labels_d[
            np.argmax(patch_prob_output[0, :], axis=-1) == i_label
        ] = i_label

    # output downsampled predicted labels
    mesh2 = mesh_d.clone()
    mesh2.celldata["Label"] = predicted_labels_d

    if not refine:
        print("Total time: ", time.time() - start_time)
        return mesh2
    else:
        # refinement
        print("\tRefining by pygco...")
        round_factor = 100
        patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6

        # unaries
        unaries = -round_factor * np.log10(patch_prob_output)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, cfg.model.num_classes)

        # parawise
        pairwise = 1 - np.eye(cfg.model.num_classes, dtype=np.int32)

        # edges
        cell_ids = np.asarray(mesh_d.faces())

        lambda_c = 30
        edges = np.empty([1, 3], order="C")
        for i_node in range(cells.shape[0]):
            # Find neighbors
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei == 2)
            for i_nei in nei_id[0][:]:
                if i_node < i_nei:
                    cos_theta = (
                        np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])
                        / np.linalg.norm(normals[i_node, 0:3])
                        / np.linalg.norm(normals[i_nei, 0:3])
                    )
                    if cos_theta >= 1.0:
                        cos_theta = 0.9999
                    theta = np.arccos(cos_theta)
                    phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                    if theta > np.pi / 2.0:
                        edges = np.concatenate(
                            (
                                edges,
                                np.array(
                                    [i_node, i_nei, -np.log10(theta / np.pi) * phi]
                                ).reshape(1, 3),
                            ),
                            axis=0,
                        )
                    else:
                        beta = 1 + np.linalg.norm(
                            np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])
                        )
                        edges = np.concatenate(
                            (
                                edges,
                                np.array(
                                    [
                                        i_node,
                                        i_nei,
                                        -beta * np.log10(theta / np.pi) * phi,
                                    ]
                                ).reshape(1, 3),
                            ),
                            axis=0,
                        )
        edges = np.delete(edges, 0, 0)
        edges[:, 2] *= lambda_c * round_factor
        edges = edges.astype(np.int32)

        refine_labels = cut_from_graph(edges, unaries, pairwise)
        refine_labels = refine_labels.reshape([-1, 1])

        # output refined result
        mesh3 = mesh_d.clone()
        mesh3.celldata["Label"] = refine_labels

        print("Total time: ", time.time() - start_time)
        return mesh3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-cfg",
        "--config_file",
        type=str,
        metavar="",
        help="configuration file",
        default="config/default.yaml",
    )
    parser.add_argument(
        "-gco",
        "--pygco",
        type=ast.literal_eval,
        metavar="",
        help="pygco",
        required=True,
    )
    parser.add_argument(
        "-ckpt", "--ckpt_path", type=str, metavar="", help="ckpt file", required=True
    )
    parser.add_argument(
        "-mesh", "--mesh_file", type=str, metavar="", help="mesh file", required=True
    )

    args = parser.parse_args()

    output = infer(args.config_file, args.ckpt_path, args.mesh_file, args.pygco)

    vedo.write(output, "predicted.vtp")
