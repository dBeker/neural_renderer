import torch
import torch.nn.functional as F


def lighting(faces, textures, intensity_ambient=0.5, intensity_directional=0.5,
             color_ambient=(1, 1, 1), color_directional=(1, 1, 1), direction=(0, 1, 0)):
    bs, nf = faces.shape[:2]
    device = faces.device

    # arguments
    color_ambient = torch.as_tensor(color_ambient, dtype=torch.float32).to(device)
    color_directional = torch.as_tensor(color_directional, dtype=torch.float32).to(device)
    direction = torch.as_tensor(direction, dtype=torch.float32).to(device)

    if color_ambient.ndimension() == 1:
        color_ambient = color_ambient[None, :].expand((bs, 3))
    if color_directional.ndimension() == 1:
        color_directional = color_directional[None, :].expand((bs, 3))
    if direction.ndimension() == 1:
        direction = direction[None, :].expand((bs, 3))

    # create light
    light = torch.zeros(bs, nf, 3, dtype=torch.float32).to(device)

    # ambient light
    if intensity_ambient != 0:
        light = light + intensity_ambient * color_ambient[:, None, :].expand(light.shape)

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        normals = F.normalize(torch.cross(v10, v12))
        normals = normals.reshape((bs, nf, 3))
        if direction.ndimension() == 2:
            direction = direction[:, None, :].expand(normals.shape)
        cos = F.relu(torch.sum(normals * direction, axis=2))
        light = (light + intensity_directional * torch.mul(
            *(color_directional[:, None, :].expand(cos[:, :, None].shape))))

    # apply
    light = light[:, :, None, None, None, :].expand(textures.shape)
    textures = textures * light
    return textures
