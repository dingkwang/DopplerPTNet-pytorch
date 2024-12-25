from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

# import pointops_cuda


import numpy as np

class KNNQuery(Function):
    @staticmethod
    def knn_query(xyz, new_xyz, offset, new_offset, nsample):
        """
        input: xyzv: (n, 4), new_xyzv: (m, 4), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        idx_list = []
        dist2_list = []
        
        # Loop over each batch
        for b in range(offset.shape[0]):
            # Get the batch range for both original points and query points
            start_n, end_n = (0 if b == 0 else offset[b-1]), offset[b]
            start_m, end_m = (0 if b == 0 else new_offset[b-1]), new_offset[b]
            
            # Extract points and query points for the current batch
            points = xyz[start_n:end_n]  # (n_batch, 3) -> Use only (x, y, z)
            query_points = new_xyz[start_m:end_m]  # (m_batch, 3)
            
            # Efficient distance computation using broadcasting or torch.cdist
            dist_matrix_squared = torch.cdist(query_points, points, p=2) ** 2  # (m_batch, n_batch)
            
            k = min(nsample, end_m - start_m)
            # Use torch.topk to find the 'nsample' smallest distances (KNN)
            dist2, knn_indices = torch.topk(dist_matrix_squared, k=k, largest=False, dim=-1)  # (m_batch, nsample)
            
            dist2 = dist2.to(xyz.device)
            knn_indices = knn_indices.to(xyz.device)
            # Adjust indices to the global index space
            knn_indices += start_n
            
            # Append results for the current batch
            idx_list.append(knn_indices)
            dist2_list.append(dist2)

        # Concatenate the results for all batches
        idx = torch.cat(idx_list, dim=0)
        dist2 = torch.cat(dist2_list, dim=0)
            
        return idx, dist2
    
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyzv: (n, 4), new_xyzv: (m, 4), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None: new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        # pointops_cuda.knnquery_cuda(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2)
        idx, dist2 = KNNQuery.knn_query(xyz, new_xyz, offset, new_offset, nsample)
        return idx, dist2


knnquery = KNNQuery.apply


def furthestsampling_py(b, xyzv, offset, new_offset, idx):
    """
    Python implementation of the furthest point sampling algorithm with 4D input (x, y, z, v).

    Args:
        b (int): Number of batches.
        xyzv (np.array): Input point cloud coordinates with velocity, shape (N, 4).
        offset (np.array): Offset indices for batches, shape (B+1,).
        new_offset (np.array): New offset indices for batches, shape (B+1,).
        tmp (np.array): Temporary distance array, shape (N,).
        idx (np.array): Array for storing indices of sampled points, shape (M,).
    
    Returns:
        idx (np.array): Indices of sampled points, shape (M,).
    """
    n = xyzv.shape[1]  # N points per batch
    m = new_offset[-1]  # Total number of sampled points
    
    i_p = 0
    for i in range(b):  # Loop over each batch
        if i == 0:
            start_n = 0
            start_m = 0
        else:
            start_n = offset[i-1]
            start_m  = new_offset[i-1] 
        end_n = offset[i]
        end_m = new_offset[i]
        # Initialize the first sampled point in the current batch
        points = xyzv[start_n:end_n]  # Points from the current batch
        num_samples = end_m - start_m
        distances = torch.full((end_n - start_n,), np.inf).to(xyzv.device)
        idx[i_p] = torch.randint(start_n, end_n, (1,))
        for _ in range(num_samples):
            # Update distances: compute distances between all points and the latest sampled point
            last_sampled_point = points[idx[i_p] - start_n]  # Use relative index
            current_distances = torch.sum((points - last_sampled_point) ** 2, axis=1)  # Squared Euclidean distance
            distances = torch.minimum(distances, current_distances)  # Keep the minimum distance for each point
            
            # Pick the point that is the farthest from the current set of sampled points
            next_sampled_idx = distances.argmax() + start_n  # Add start_n to get the global index
            i_p+=1
            if i_p >= idx.shape[0]:
                print(i_p)
                continue
            idx[i_p] = next_sampled_idx
            
    return idx

class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyzv, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyzv.is_contiguous()
        n, b, n_max = xyzv.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        # pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        furthestsampling_py(b, xyzv, offset, new_offset, idx)
        return idx

furthestsampling = FurthestSampling.apply


import numpy as np



def gather_operation(features, idx):
    """
    Simulates gather_operation to gather features of specific points from a larger set.
    
    Args:
    - features: (B, C, N) tensor where B is batch size, C is number of channels, N is number of points.
    - idx: (B, npoint) tensor containing the indices of points to gather.
    
    Returns:
    - gathered_features: (B, C, npoint) gathered features from the input based on idx.
    """
    # Use torch.gather to gather the features according to idx
    B, C, N = features.shape  # Batch size, Feature dimension, Number of points
    _, npoint = idx.shape     # npoint is the number of points we want to gather
    
    # Expand idx to match the feature dimensions
    idx_expanded = idx.unsqueeze(1).expand(B, C, npoint)  # (B, C, npoint)
    
    # Gather the features using advanced indexing
    gathered_features = torch.gather(features, 2, idx_expanded)  # Gather along the third dimension (points)
    
    return gathered_features



def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, feature_only=True):
    """
    input: xyz: (n, 4), 
    new_xyz: (m, 4), 
    feat: (n, c), 
    idx: (m, nsample), 
    offset: (b), 
    new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset) # (m, nsample)

    c_xyz = xyz.shape[1]
    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :c_xyz].view(m, nsample, c_xyz) # (m, nsample, c_xyz)
    #grouped_xyz = grouping(xyz, idx) # (m, nsample, c_xyz)
    grouped_xyz -= new_xyz.unsqueeze(1) # (m, nsample, c_xyz)
    grouped_feat = feat[idx.view(-1).long()].view(m, nsample, c) # (m, nsample, c)
    #grouped_feat = grouping(feat, idx) # (m, nsample, c)

    if not feature_only:
        return torch.cat((grouped_xyz, grouped_feat), -1) # (m, nsample, 3+c)
    else:
        return grouped_feat


class Subtraction(Function):
    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        input: input1: (n, c), input2: (n, c), idx: (n, nsample)
        output:  (n, nsample, c)
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape; nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        pointops_cuda.subtraction_forward_cuda(n, nsample, c, input1, input2, idx, output)
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, nsample, c)
        output: grad_input1: (n, c), grad_input2: (n, c)
        """
        idx, = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.subtraction_backward_cuda(n, nsample, c, idx, grad_output, grad_input1, grad_input2)
        return grad_input1, grad_input2, None

subtraction = Subtraction.apply


class Aggregation(Function):
    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx: (n, nsample)
        output: (n, c)
        """
        assert input.is_contiguous() and position.is_contiguous() and weight.is_contiguous()
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.aggregation_forward_cuda(n, nsample, c, w_c, input, position, weight, idx, output)
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, c)
        output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight : (n, nsample, c')
        """
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape; w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        pointops_cuda.aggregation_backward_cuda(n, nsample, c, w_c, input, position, weight, idx, grad_output, grad_input, grad_position, grad_weight)
        return grad_input, grad_position, grad_weight, None

aggregation = Aggregation.apply


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)
    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, k), (n, k)
        dist_recip = 1.0 / (dist + 1e-8) # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm # (n, k)

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(m, c).zero_()
        pointops_cuda.interpolation_backward_cuda(n, c, k, grad_output, idx, weight, grad_input)
        return None, None, grad_input, None, None, None

interpolation2 = Interpolation.apply
