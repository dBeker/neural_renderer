import torch
import torch.nn.functional as F


def look(vertices, eye, direction=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndimension() != 3)

    device = vertices.device

    if direction is None:
        direction = torch.as_tensor([0, 0, 1], dtype=torch.float32, device=device)
    if up is None:
        up = torch.as_tensor([0, 1, 0], dtype=torch.float32, device=device)

    eye = torch.as_tensor(eye, dtype=torch.float32, device=device)

    if eye.ndimension() == 1:
        eye = eye[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]
    if up.ndimension() == 1:
        up = up[None, :]

    # create new axes
    z_axis = F.normalize(direction)
    x_axis = F.normalize(torch.cross(up, z_axis))
    y_axis = F.normalize(torch.cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    r = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    if r.shape[0] != vertices.shape[0]:
        r = r.expand(vertices.shape)

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = eye[:, None, :].expand(vertices.shape)
    vertices = vertices - eye
    vertices = torch.matmul(vertices, r.transpose(1, 2))

    return vertices
