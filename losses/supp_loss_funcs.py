import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from typing import Union


EPS = 1e-10

##############################################################################
# Classes
##############################################################################
class RotationLoss(nn.Module):
    """Define Rotation loss between non-orthogonal matrices.
    """

    def __init__(self, device, which_metric='MSE'):
        super(RotationLoss, self).__init__()
        self.device = device
        self.indentity_rot = torch.eye(3, device=self.device).unsqueeze(0)

        self.which_metric = which_metric
        assert self.which_metric in ['cosine', 'angular', 'orthogonal', 'MSE'], f'{self.which_metric} invalid rot loss'

        if self.which_metric == 'cosine':
            self.metric = torch.nn.CosineSimilarity(dim=2)
        else:
            self.metric = torch.nn.MSELoss()

    def batched_trace(self, mat):
        return mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    def cosine_loss(self, R1, R2):
        return torch.mean(1 - self.metric(R1, R2))

    def angular_loss(self, R1, R2):
        M = torch.matmul(R1, R2.transpose(1, 2))
        return torch.mean(torch.acos(torch.clamp((self.batched_trace(M) - 1) / 2., -1 + EPS, 1 - EPS)))

    def __call__(self, R1, R2):
        # Input:
        #    R1, R2 - Bx3x3 (dim=1 - channels, dim=2 - xyz)
        # Output:
        #    loss - torch tensor

        if self.which_metric == 'cosine':
            return self.cosine_loss(R1, R2)
        elif self.which_metric == 'angular':
            return self.angular_loss(R1, R2)
        elif self.which_metric == 'orthogonal':
            return self.criterionMSE(torch.matmul(R1, R2.transpose(1, 2)),
                                     self.indentity_rot.expand(R1.size(0), 3, 3))
        else:
            return self.metric(R1, R2)


class DirichletLoss(nn.Module):
    """Define symmetric dirichlet loss.
    """

    def __init__(self, device):
        super(DirichletLoss, self).__init__()
        self.device = device

    def __call__(self, R):
        RTR = torch.matmul(R.transpose(1, 2), R)
        RTRinv = torch.inverse(RTR)

        # Explicit:
        # a = RTR[:, 0, 0]
        # b = RTR[:, 0, 1]
        # c = RTR[:, 0, 2]
        # d = RTR[:, 1, 0]
        # e = RTR[:, 1, 1]
        # f = RTR[:, 1, 2]
        # g = RTR[:, 2, 0]
        # h = RTR[:, 2, 1]
        # i = RTR[:, 2, 2]
        #
        # RTRinv = torch.zeros_like(RTR, device=self.device)
        # RTRinv[:, 0, 0] = e * i - f * h
        # RTRinv[:, 0, 1] = -1 * (d * i -f * g)
        # RTRinv[:, 0, 2] = d * h - g * e
        # RTRinv[:, 1, 0] = -1 * (b * i - c * h)
        # RTRinv[:, 1, 1] = a * i - c * g
        # RTRinv[:, 1, 2] = -1 * (a * h - b * g)
        # RTRinv[:, 2, 0] = f * b - c * e
        # RTRinv[:, 2, 1] = -1 * (a * f - c * d)
        # RTRinv[:, 2, 2] = a * e - b * d
        #
        # det = a * RTRinv[:, 0, 0] + b * RTRinv[:, 0, 1] + c * RTRinv[:, 0, 2] + 1e-15
        # RTRinv = RTRinv.transpose(1,2) / det.unsqueeze(-1).unsqueeze(-1)

        rigidity_loss = (RTR ** 2).sum(1).sum(1).sqrt() + (RTRinv ** 2).sum(1).sum(1).sqrt()

        return rigidity_loss.mean()


class OrthogonalLoss(nn.Module):
    """Define orthogonal loss for non-orthogonal matrix.
    """

    def __init__(self, device, which_metric='MSE'):
        super(OrthogonalLoss, self).__init__()
        self.device = device
        self.indentity_rot = torch.eye(3, device=self.device).unsqueeze(0)

        self.which_metric = which_metric
        assert self.which_metric in ['dirichlet', 'svd', 'MSE'], f'{self.which_metric} invalid ortho loss'

        if self.which_metric == 'dirichlet':
            self.metric = self.DirichletLoss(self.device)
        else:
            self.metric = torch.nn.MSELoss()

    def __call__(self, R1):
        # Input:
        #    R1, R2 - Bx3x3 (dim=1 - channels, dim=2 - xyz)
        # Output:
        #    loss - torch tensor

        if self.which_metric == 'dirichlet':
            return self.metric(R1)
        elif self.which_metric == 'svd':
            u, s, v = torch.svd(R1)
            return self.metric(R1, torch.matmul(u, v.transpose(1, 2)))
        else:
            return self.metric(torch.matmul(R1, R1.transpose(1, 2)),
                               self.indentity_rot.expand(R1.size(0), 3,
                                                         3))

def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def u_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.
    x is partial, y is full
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    #  find in y, for each point in x
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_x, cham_y, cham_normals