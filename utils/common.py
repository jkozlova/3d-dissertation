import copy
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision
from clearml import Task
from pytorch3d.structures.meshes import Meshes

from .HeadNeRF.HeadNeRFOptions import BaseOptions


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


@dataclass
class FlameDatasetOptions:
    n_shape: int = 100
    n_expr: int = 100
    n_tex: int = 100
    tex_res: int = 512

    ignore_lower_neck: bool = True

    device: str = "cuda"


class FlameRenderer:
    def __init__(
        self,
        image_size,
        options: FlameDatasetOptions = FlameDatasetOptions(),
        use_mouth: bool = False,
        tex_mouth_path: str = "",
    ) -> None:
        self._image_size = image_size
        self.options = options
        self._flame = FlameHead(self.options.n_shape, self.options.n_expr)
        self._flame.to(self.options.device)

        self._flame_tex = FlameTex(
            self.options.n_tex, use_mouth=use_mouth, tex_mouth_path=tex_mouth_path
        )
        self._flame_tex.to(self.options.device)

        self._render = SHRenderer()

        if self.options.ignore_lower_neck:
            self._face_mask = torch.ones(
                len(self._flame.faces), device=self.options.device
            ).bool()
            lower_neck_ids = np.load(FLAME_LOWER_NECK_FACES_PATH)
            self._face_mask[lower_neck_ids] = 0

    def render(self, params: dict, i: Optional[int]) -> torch.Tensor:
        """
        Render a flame model given parameters.

        Args:
            params (dict): A dictionary containing parameters of the flame model.
            i (int or None): Index of the frame to render. If None, render the first frame.

        Returns:
            torch.Tensor: A rendered image.
        """
        params = copy.deepcopy(params)

        if i is None:
            verts, _, albedos = self._forward_flame(
                params["shape"].unsqueeze(0),
                params["expr"].unsqueeze(0),
                params["neck_pose"].unsqueeze(0),
                params["jaw_pose"].unsqueeze(0),
                params["eyes_pose"].unsqueeze(0),
                params["texture"].unsqueeze(0),
            )
            image, _ = self._visualize_tracking(
                verts,
                albedos,
                params["K"].unsqueeze(0),
                params["RT"].unsqueeze(0),
                params["light"],
            )
        else:
            verts, _, albedos = self._forward_flame(
                params["shape"][i].unsqueeze(0),
                params["expr"][i].unsqueeze(0),
                params["neck_pose"][i].unsqueeze(0),
                params["jaw_pose"][i].unsqueeze(0),
                params["eyes_pose"][i].unsqueeze(0),
                params["texture"][i].unsqueeze(0),
            )
            image, _ = self._visualize_tracking(
                verts,
                albedos,
                params["K"][i].unsqueeze(0),
                params["RT"][i].unsqueeze(0),
                params["light"][i],
            )
        return torchvision.transforms.functional.hflip(image)

    @torch.no_grad()
    def _visualize_tracking(
        self,
        vertices: torch.Tensor,
        albedos: torch.Tensor,
        K: torch.Tensor,
        RT: torch.Tensor,
        lights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Visualize tracking results using the given parameters.

        Args:
            vertices (torch.Tensor): The vertices of the 3D model.
            albedos (torch.Tensor): Albedo values for the model.
            K (torch.Tensor): Camera intrinsic matrix.
            RT (torch.Tensor): Rotation and translation matrix.
            lights (torch.Tensor): Lighting information.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the predicted image
            and the alpha channel as torch.Tensors.
        """
        rasterization_results = self._rasterize_flame(
            K, RT, vertices, scale=1, use_cache=False
        )
        render_result = self._render_rgba(rasterization_results, albedos, lights)
        predicted_images, alpha = render_result[:, :3], render_result[:, 3:]
        predicted_images = torch.clip(predicted_images, min=0, max=1)
        predicted_images = predicted_images * alpha
        return predicted_images[0], alpha[0, 0, ...]  # , mask

    def _forward_flame(
        self,
        shape: torch.Tensor,
        expr: torch.Tensor,
        neck_pose: torch.Tensor,
        jaw_pose: torch.Tensor,
        eyes_pose: torch.Tensor,
        texture: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the flame model using the given parameters

        Args:
            shape (torch.Tensor): A tensor of shape (1, n_shape) containing the shape parameters.
            expr (torch.Tensor): A tensor of shape (1, n_expr) containing the expression parameters.
            neck_pose (torch.Tensor): A tensor of shape (1, n_neck) containing the neck pose parameters.
            jaw_pose (torch.Tensor): A tensor of shape (1, n_jaw) containing the jaw pose parameters.
            eyes_pose (torch.Tensor): A tensor of shape (1, n_eyes) containing the eyes pose parameters.
            texture (torch.Tensor): A tensor of shape (1, n_tex) containing the texture parameters.

        Returns:
            tuple: A tuple of four tensors: vertices (n_vertices, 3), landmarks (n_landmarks, 2),
            albedos (n_vertices, 3), and the rendered image (height, width, 3)
        """
        ret = self._flame(
            shape,
            expr,
            neck_pose,
            jaw_pose,
            eyes_pose,
        )

        verts, lmks = ret[0], ret[1]
        albedos = self._flame_tex(texture)
        return verts, lmks, albedos

    def _rasterize_flame(
        self,
        K: torch.Tensor,
        RT: torch.Tensor,
        vertices: torch.Tensor,
        scale: float = 1.0,
        use_cache: bool = True,
    ) -> dict:
        faces = self._flame.faces
        if self.options.ignore_lower_neck:
            faces = faces[self._face_mask]

        H, W = self._image_size
        H, W = int(H * scale), int(W * scale)
        K = K * scale

        cameras = create_camera_objects(K, RT, (H, W), self.options.device)

        flame_meshes = Meshes(verts=vertices, faces=faces.expand(len(vertices), -1, -1))
        render_result = self._render.rasterize(flame_meshes, cameras, (H, W), use_cache)

        return {
            "fragments": render_result[0],
            "screen_coords": render_result[1],
            "meshes": flame_meshes,
        }

    def _render_rgba(
        self, rasterization_result: dict, albedos: torch.Tensor, lights: torch.Tensor
    ) -> torch.Tensor:
        """
        Render the rgba image from the rasterization result and the optimized texture + lights
        Args:
            rasterization_result (dict): A dictionary containing the rasterization result.
            albedos (torch.Tensor): A tensor of shape (n_vertices, 3) containing the albedo values.
            lights (torch.Tensor): A tensor of shape (1, 9) containing the lights parameters.
        Returns:
            torch.Tensor: A tensor of shape (height, width, 4) containing the rendered image.
        """
        fragments = rasterization_result["fragments"]
        meshes = rasterization_result["meshes"]

        B = len(meshes)
        lights = lights[None].expand(B, -1, -1)
        uv_coords = self._flame.face_uvcoords
        if self.options.ignore_lower_neck:
            uv_coords = uv_coords[self._face_mask]
        uv_coords = uv_coords.repeat(B, 1, 1)
        return self._render.render_rgba(meshes, fragments, uv_coords, albedos, lights)


def get_rays(
    options: BaseOptions, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates rays for rendering a batch of images.

    Args:
    - options (BaseOptions): A BaseOptions object containing the featmap_size.
    - batch_size (int): The batch size for which to generate rays.

    Returns:
    - A tuple of two tensors, ray_xy and ray_uv, representing the 2D coordinates
      of the rays in image space and normalized uv space, respectively. The
      tensors have shape (batch_size, 2, featmap_size, featmap_size) and
      (batch_size, featmap_size, featmap_size, 2), respectively.
    """
    mini_h = options.featmap_size
    mini_w = options.featmap_size

    indexs = torch.arange(mini_h * mini_w)
    x_coor = (indexs % mini_w).view(-1)
    y_coor = torch.div(indexs, mini_w, rounding_mode="floor").view(-1)

    xy = torch.stack([x_coor, y_coor], dim=0).float()
    uv = torch.stack(
        [x_coor.float() / float(mini_w), y_coor.float() / float(mini_h)], dim=-1
    )

    ray_xy = xy.repeat(batch_size, 1, 1)
    ray_uv = uv.repeat(batch_size, 1, 1)

    return ray_xy, ray_uv


def get_data(
    data: Dict[str, torch.Tensor],
    options: BaseOptions,
    offsets: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process data and return relevant tensors for neural network inputs.

    Args:
    - data (Dict[str, torch.Tensor]): A dictionary containing data tensors.
    - options (BaseOptions): An instance of BaseOptions containing configuration options.
    - offsets (Optional[Dict[str, torch.Tensor]]): An optional dictionary containing offset tensors (default is None).

    Returns:
    - A tuple of torch.Tensors containing shape_code, appea_code, batch_Rmats, batch_Tvecs, temp_inv_inmat.
    """

    data_copy = copy.deepcopy(data)
    if offsets is not None:
        shape_code = torch.hstack(
            [
                data_copy["params"]["shape"] + offsets["shape_offset"],
                data_copy["params"]["expr"] + offsets["expr_offset"],
                data_copy["params"]["neck_pose"] + offsets["pose_neck_offset"],
                data_copy["params"]["jaw_pose"] + offsets["pose_jaw_offset"],
                data_copy["params"]["eyes_pose"],
            ]
        )
        appea_code = (
            torch.hstack(
                [
                    data_copy["params"]["texture"],
                    data_copy["params"]["light"].flatten(start_dim=1),
                ]
            )
            + offsets["apea_offset"]
        )
    else:
        shape_code = torch.hstack(
            [
                data_copy["params"]["shape"][:, :100],
                data_copy["params"]["expr"],
                data_copy["params"]["neck_pose"],
                data_copy["params"]["jaw_pose"],
                data_copy["params"]["eyes_pose"],
            ]
        )
        appea_code = torch.hstack(
            [
                data_copy["params"]["texture"],
                data_copy["params"]["light"].flatten(start_dim=1),
            ]
        )
    batch_c2w_rotation_matrices = data_copy["params"]["RT"][:, :, :3]
    batch_c2w_translation_vectors = data_copy["params"]["RT"][:, :, 3]
    batch_rt_rotation_matrices = batch_c2w_rotation_matrices.transpose(1, 2)
    batch_rt_translation_vectors = -torch.matmul(
        batch_rt_rotation_matrices, batch_c2w_translation_vectors.unsqueeze(-1)
    ).squeeze(-1)

    data_copy["params"]["RT"] = torch.cat(
        [
            batch_rt_rotation_matrices,
            batch_rt_translation_vectors.unsqueeze(-1),
        ],
        dim=-1,
    )

    batch_Rmats = data_copy["params"]["RT"][:, :, :3]
    batch_Tvecs = data_copy["params"]["RT"][:, :, 3]

    temp_inmat = data_copy["params"]["K"].clone()
    temp_inmat[:, :2, :] *= options.featmap_size / options.pred_img_size
    temp_inv_inmat = torch.linalg.inv(temp_inmat)

    return shape_code, appea_code, batch_Rmats, batch_Tvecs, temp_inv_inmat


def setup_random_seeds() -> None:
    """
    Set random seeds for PyTorch, CUDA and NumPy, respectively, to ensure reproducibility.
    """
    torch.manual_seed(35)
    torch.cuda.manual_seed(64)
    np.random.seed(42)


def create_logger(
    parameters: Dict[str, Any]
) -> Tuple[torch.utils.tensorboard.SummaryWriter, Task, str]:
    """
    Create a logger for tensorboard and a clearml task.

    Args:
    - parameters (Dict[str, Any]): A dictionary containing parameters such as mode, name, num_samle_coarse, pc_name, losses, and disentangled_loss.

    Returns:
    - A tuple containing a SummaryWriter, a clearml Task, and a string representing the logdir path.
    """
    dt = datetime.now()
    project_name = f"Dissertation - Head Avatar/{parameters['mode'].capitalize()}"

    losses = ""
    for key, value in parameters["losses"].items():
        losses += f"{key}-{value}_"
    task_name = f"{parameters['name']}_{'disloss' if parameters['disentangled_loss'] else ''}_{parameters['num_samle_coarse']}pt_{parameters['pc_name']}_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}"
    logdir_path = os.path.join(project_name, task_name)
    os.makedirs(logdir_path)
    task = Task.init(project_name=project_name, task_name=task_name)
    writer = SummaryWriter(logdir_path)
    return writer, task.connect(parameters), logdir_path
