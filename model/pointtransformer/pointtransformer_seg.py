from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pointops.functions import pointops_cuda as pointops

class PointnetFPModule(nn.Module):

    def __init__(self, in_planes, out_planes=None, device='cuda'):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), 
                                         nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True)).to(device)
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), 
                                         nn.ReLU(inplace=True)).to(device)
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), 
                                         nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True)).to(device)
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), 
                                         nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True)).to(device)

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b_sum = x_b.sum(0, True) / cnt
                assert self.linear2[0].in_features == x_b_sum.shape[-1]
                project_x = self.linear2(x_b_sum).repeat(cnt, 1)
                x_b = torch.cat((x_b, project_x), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            assert self.linear1[0].in_features == x.shape[-1]
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            assert self.linear2[0].in_features == x2.shape[-1]
            proj_x2 = self.linear2(x2)
            
            interpolated = pointops.interpolation(p2, p1, proj_x2, o2, o1)
            assert self.linear1[0].in_features == x1.shape[-1]
            assert interpolated.shape == x1.shape
            x = self.linear1(x1) + interpolated
        return x


class DopplerPTNet(nn.Module):
    def __init__(self, num_classes, input_channels=4, use_xyz=True, device='cuda'):
        super(DopplerPTNet, self).__init__()
        self.device = device
        self.in_planes, planes = input_channels, [64, 128, 256, 512]
        strides, nsamples = [1, 4, 4, 4], [16, 16, 16, 16]

        self.num_classes = num_classes
        self.use_xyz = use_xyz

        # Encoder (Set Abstraction modules)

        self.sa1 = PointNetSetAbstractionPN2(in_planes=input_channels,
                                             out_planes=planes[0],
                                             stride=strides[0],
                                             nsample=nsamples[0],
                                             device=device)
        self.sa2 = PointNetSetAbstractionPN2(in_planes=planes[0],
                                             out_planes=planes[1],
                                             stride=strides[1],
                                             nsample=nsamples[1],
                                             device=device)
        self.sa3 = PointNetSetAbstractionPN2(in_planes=planes[1],
                                             out_planes=planes[2],
                                             stride=strides[2],
                                             nsample=nsamples[2],
                                             device=device)
        self.sa4 = PointNetSetAbstractionPN2(in_planes=planes[2],
                                             out_planes=planes[3],
                                             stride=strides[3],
                                             nsample=nsamples[3],
                                             device=device)

        # Point Transformer layers
        self.pt1 = PointTransformerLayer(planes[0], share_planes=8)
        self.pt2 = PointTransformerLayer(planes[1], share_planes=8)
        self.pt3 = PointTransformerLayer(planes[2], share_planes=8)
        self.pt4 = PointTransformerLayer(planes[3], share_planes=8)

        # Decoder (Feature Propagation modules)
        self.fp4 = PointnetFPModule(planes[3], out_planes = None)
        self.fp3 = PointnetFPModule(in_planes=planes[3], out_planes=planes[2])
        self.fp2 = PointnetFPModule(in_planes=planes[2], out_planes=planes[1])
        self.fp1 = PointnetFPModule(in_planes=planes[1], out_planes=planes[0])

        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]).to(self.device),
            nn.BatchNorm1d(planes[0]).to(self.device),  # Add Batch Normalization
            nn.ReLU(inplace=True).to(self.device),  # Add ReLU Activation
            nn.Dropout(0.1).to(self.device),  # Dropout layer to prevent overfitting
            nn.Linear(planes[0], num_classes).to(self.device))

    def forward_2(self, xyz, feat, offset):

        # Encoder
        l1_xyz, l1_features, offset1, l1_velocity = self.sa1(xyz, feat, offset)
        # l1_features: (N/4, mid_planes)
        l1_features = self.pt1(l1_xyz, l1_features, offset1, l1_velocity)  # (N/4, 64)
        # l1_features: (N/4, out_planes)

        l2_xyz, l2_features, offset2, l2_velocity = self.sa2(l1_xyz, l1_features, offset1)
        l2_features = self.pt2(l2_xyz, l2_features, offset2, l2_velocity)  # (N/8, 128)

        l3_xyz, l3_features, offset3, l3_velocity = self.sa3(l2_xyz, l2_features, offset2)
        l3_features = self.pt3(l3_xyz, l3_features, offset3, l3_velocity)  # (N/4, 256)

        l4_xyz, l4_features, offset4, l4_velocity = self.sa4(l3_xyz, l3_features, offset3)
        l4_features = self.pt4(l4_xyz, l4_features, offset4, l4_velocity)  # (N/16, 512)

        # Decoder
        l4_features = self.fp4(pxo1=[l4_xyz, l4_features, offset4], pxo2=None)  # ( , 512)
        l3_features = self.fp3(pxo1=[l3_xyz, l3_features, offset3], pxo2=[l4_xyz, l4_features, offset4])  # (N/16, 256)
        l2_features = self.fp2(pxo1=[l2_xyz, l2_features, offset2], pxo2=[l3_xyz, l3_features, offset3])  # (N/8, 128)
        l1_features = self.fp1(pxo1=[l1_xyz, l1_features, offset1], pxo2=[l2_xyz, l2_features, offset2])  # l1_features.shapes [N, 64] -> 

        # Final fully connected layers for point-wise classification
        logits = self.cls(l1_features)  # l1_features: (N/4, 64)
        return logits
    
    def forward(self, xyz, feat, offset):

        # Encoder
        l1_xyz, l1_features, offset1, l1_velocity = self.sa1(xyz, feat, offset, feat)
        # # l1_features: (N/4, mid_planes)
        l1_features = self.pt1(l1_xyz, l1_features, offset1, l1_velocity)  # (N/4, 64)
        # l1_features: (N/4, out_planes)

        l2_xyz, l2_features, offset2, l2_velocity = self.sa2(l1_xyz, l1_features, offset1, l1_velocity)
        l2_features = self.pt2(l2_xyz, l2_features, offset2, l2_velocity)  # (N/8, 128)

        l3_xyz, l3_features, offset3, l3_velocity = self.sa3(l2_xyz, l2_features, offset2, l2_velocity)
        l3_features = self.pt3(l3_xyz, l3_features, offset3, l3_velocity)  # (N/4, 256)

        l4_xyz, l4_features, offset4, l4_velocity = self.sa4(l3_xyz, l3_features, offset3, l3_velocity)
        l4_features = self.pt4(l4_xyz, l4_features, offset4, l4_velocity)  # (N/16, 512)

        # # Decoder
        l4_features = self.fp4(pxo1=[l4_xyz, l4_features, offset4], pxo2=None)  # ( , 512)
        l3_features = self.fp3(pxo1=[l3_xyz, l3_features, offset3], pxo2=[l4_xyz, l4_features, offset4])  # (N/16, 256)
        l2_features = self.fp2(pxo1=[l2_xyz, l2_features, offset2], pxo2=[l3_xyz, l3_features, offset3])  # (N/8, 128)
        l1_features = self.fp1(pxo1=[l1_xyz, l1_features, offset1], pxo2=[l2_xyz, l2_features, offset2])  # l1_features.shapes [N, 64] -> 
        # # Final fully connected layers for point-wise classification
        logits = self.cls(l1_features)  # l1_features: (N/4, 64)
        return logits
    
    def pred(self, logits):
        # Convert logits to probabilities
        pred = torch.argmax(logits, dim=1)
        return pred

    def compute_loss(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        return loss

# RoI Pooling placeholder (this should pool features from the proposed regions)
class RoIPooling(nn.Module):

    def __init__(self):
        super(RoIPooling, self).__init__()
        # Placeholder for actual implementation
        # Define pooling operations for 3D RoI pooling

    def forward(self, box_proposals, features):
        # Placeholder logic for RoI Pooling over proposed regions
        # Extract features from each region of interest (RoI) based on box proposals
        # Return pooled features
        return features  # Modify this based on RoI pooling logic



class RoIPooling3D(nn.Module):

    def __init__(self, pooled_size=2, device='cuda'):
        """
        RoI Pooling layer for 3D point cloud.
        Parameters:
        - pooled_size: the size to which features are pooled for each RoI (e.g., 2x2x2 grid).
        """
        super(RoIPooling3D, self).__init__()
        self.pooled_size = pooled_size

    def forward(self, box_proposals, point_features, offset, point_cloud):
        """
        Parameters:
        - box_proposals: (B, N, 6) bounding box proposals (x, y, z, w, l, h).
        - point_features: (B, C, P) features corresponding to each point in the point cloud.
        - offset: (B) offset for each batch.
        - point_cloud: (B, P, 3) coordinates of the point cloud (x, y, z).
        
        Returns:
        - pooled_features: (B, N, C, pooled_size, pooled_size, pooled_size)
        """

        _, num_boxes, _ = box_proposals.shape
        pooled_features = []

        for i, o in enumerate(offset):
            batch_pooled_features = []
            for box_idx in range(num_boxes):
                # Get the current box proposal (x, y, z, w, l, h)
                box = box_proposals[i, box_idx]
                center = box[:3]  # (x, y, z) center of the box
                size = box[3:6]  # (w, l, h) size of the box

                # Find points inside the box
                mask = self._get_points_in_box(point_cloud[:o], center, size)
                selected_points = point_features[i, :, mask]

                if selected_points.shape[1] == 0:
                    # If no points are found in the box, skip
                    pooled = torch.zeros(
                        (point_features.shape[1], self.pooled_size, self.pooled_size, self.pooled_size),
                        device=point_features.device)
                else:
                    # Pooling: Apply pooling on selected points (e.g., max-pooling)
                    pooled = self._pool_features(selected_points, self.pooled_size)

                batch_pooled_features.append(pooled)

            # Stack all pooled features for the batch
            batch_pooled_features = torch.stack(batch_pooled_features, dim=0)
            pooled_features.append(batch_pooled_features)

        # Return pooled features for the batch
        return torch.stack(pooled_features, dim=0)

    def _get_points_in_box(self, points, center, size):
        """
        Helper function to find points inside a given box.
        Parameters:
        - points: (P, 3) point cloud (x, y, z) for one batch.
        - center: (3) center of the box (x, y, z).
        - size: (3) size of the box (w, l, h).
        
        Returns:
        - mask: (P) boolean mask indicating points inside the box.
        """
        lower_bound = center - size / 2
        upper_bound = center + size / 2

        mask = ((points[:, 0] >= lower_bound[0]) & (points[:, 0] <= upper_bound[0]) & (points[:, 1] >= lower_bound[1]) &
                (points[:, 1] <= upper_bound[1]) & (points[:, 2] >= lower_bound[2]) & (points[:, 2] <= upper_bound[2]))

        return mask

    def _pool_features(self, features, pooled_size):
        """
        Pool the features into a fixed size grid (e.g., max-pooling).
        Parameters:
        - features: (C, P_selected) selected features for points in the box.
        - pooled_size: the size to pool to (e.g., 2x2x2 grid).
        
        Returns:
        - pooled: (C, pooled_size, pooled_size, pooled_size) pooled features.
        """
        # Assuming that the features represent points evenly distributed in the box,
        # we can use max-pooling here. You could also use average pooling.
        pooled = F.adaptive_max_pool1d(features.unsqueeze(0), pooled_size**3)  # Pool to pooled_size^3
        pooled = pooled.view(features.size(0), pooled_size, pooled_size, pooled_size)  # Reshape to 3D

        return pooled


class PointTransformerLayer(nn.Module):

    def __init__(self, in_planes, share_planes=8, nsample=16, device='cuda'):
        super().__init__()
        self.mid_planes = mid_planes = out_planes = in_planes
        self.out_planes = in_planes
        self.share_planes = share_planes
        self.nsample = nsample

        # Learnable transformations for query (q), key (k), and value (v)
        self.linear_q = nn.Linear(in_planes, in_planes).to(device)
        self.linear_k = nn.Linear(in_planes, in_planes).to(device)
        self.linear_v = nn.Linear(in_planes, in_planes).to(device)

        # Positional encoding
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3).to(device),
            nn.BatchNorm1d(3).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.Linear(3, in_planes).to(device))

        # Velocity encoding (for Doppler or similar velocity features)
        self.linear_velocity_encoding = nn.Sequential(
            nn.Linear(1, 1).to(device),
            nn.BatchNorm1d(1).to(device),
            nn.ReLU(inplace=True).to(device),
            nn.Linear(1, out_planes).to(device))

        # Attention weight processing
        assert mid_planes // share_planes == out_planes // share_planes
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes)).to(device)

        # Final transformation rho after combining position and velocity
        self.rho = nn.Sequential(nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True), nn.Linear(out_planes,
                                                                                              out_planes)).to(device)

        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, xyz, features, offset, velocities) -> torch.Tensor:
        # xyz contains both coordinates (x, y, z) and velocity (v)
        coords = xyz  # Extract spatial coordinates (x, y, z)

        # Step 1: Transform input features into query (q), key (k), value (v)
        assert self.linear_k.in_features == features.shape[-1]
        x_q = self.linear_q(features)  # Query: (N, mid_planes)
        x_k = self.linear_k(features)  # Key: (N, mid_planes)
        x_v = self.linear_v(features)  # Value: (N, out_planes=128)

        # Step 2: Grouping using pointops (get neighbors for each point)
        x_k_grouped = pointops.queryandgroup(self.nsample,
                                             coords,
                                             coords,
                                             x_k,
                                             None,
                                             offset,
                                             offset,
                                             feature_only=False)  # Group neighbors for key
        # (M, nsample, mid_planes)
        x_v_grouped = pointops.queryandgroup(self.nsample, coords, coords, x_v, None, offset, offset,
                                             feature_only=True)  
        # Group neighbors for value (M, nsample, out_planes)

        # Step 3: Positional encoding (δ)
        p_r, x_k_grouped = x_k_grouped[:, :, :3], x_k_grouped[:, :, 3:]  # Separate positions
        assert p_r.shape[-1] == self.linear_p[0].in_features
        p_r_encoded = self._apply_mlp(p_r, self.linear_p)  # Apply position encoding

        # Step 4: Velocity encoding (β)
        v_r_grouped = pointops.queryandgroup(nsample=self.nsample,
                                             xyz=velocities,
                                             new_xyz=velocities,
                                             feat=velocities,
                                             idx=None,
                                             offset=offset,
                                             new_offset=offset,
                                             feature_only=True)  
        # Get velocity differences (M, nsample, 1)
        assert v_r_grouped.shape[-1] == self.linear_velocity_encoding[0].in_features
        v_r_encoded = self._apply_mlp(v_r_grouped, self.linear_velocity_encoding)  # Apply velocity encoding
        # (M, nsample, out_planes)

        # Step 5: Combine (ϕ(x_i) - ψ(x_j)) + δ + β substraction vector attention
        w = (x_k_grouped - x_q.unsqueeze(1)) + (p_r_encoded + v_r_encoded)  
        # Combine query-key differences with positional and velocity encodings

        # Step 6: Process attention weights
        assert w.shape[-1] == self.linear_w[2].in_features
        w = self._apply_mlp(w, self.linear_w)  # Apply attention weight transformations
        w = self.softmax(w)  # Softmax over attention weights

        # Step 7: Apply attention weights to value vectors
        n, nsample, c = x_v_grouped.shape
        s = self.share_planes
        output = ((x_v_grouped + p_r_encoded + v_r_encoded).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(
            n, c)
        # (M, out_planes)
        output += features

        # Step 8: Apply the final transformation (rho)
        assert output.shape[-1] == self.rho[-1].in_features
        output = self.rho(output)
        return output

    def _apply_mlp(self, input_tensor, mlp_layer):
        """
        Applies an MLP layer to the input tensor, handling batch and transpose for 1D batch normalization.
        """
        for i, layer in enumerate(mlp_layer):
            if isinstance(layer, nn.BatchNorm1d):
                input_tensor = input_tensor.transpose(1, 2).contiguous()
                input_tensor = layer(input_tensor)
                input_tensor = input_tensor.transpose(1, 2).contiguous()
            else:
                input_tensor = layer(input_tensor)
        return input_tensor


class PointNetSetAbstractionPN2(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, nsample=16, device='cuda'):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(in_planes, out_planes, bias=False).to(device)
            self.pool = nn.MaxPool1d(nsample).to(device)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False).to(device)
        self.bn = nn.BatchNorm1d(out_planes).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)

    def forward(self, xyz, feat, offset, velocities):
        if self.stride != 1: # sa2, sa3, sa4
            stride_tensor = torch.tensor(self.stride, device=offset.device)
            # count = torch.floor_divide(offset[0], stride_tensor)
            count  = torch.div(offset[0], stride_tensor, rounding_mode='trunc')
            new_offset = [count.clone()]  # Clone the tensor to ensure it's a separate copy
            for i in range(1, offset.shape[0]):
                count += (offset[i].item() - offset[i - 1].item()) // self.stride
                new_offset.append(count.clone())  # Clone the tensor before appending
            new_offset = torch.cuda.IntTensor(new_offset)
            xyzv = torch.cat([xyz, feat], dim=1)
            idx = pointops.furthestsampling(xyzv, offset,
                                            new_offset)  # (m)  input: xyz: (n, 4), offset: (b), new_offset: (b)
            new_xyz = xyz[idx.long(), :]  # (m, 3)
            velocities = velocities[idx.long(), :]  # (m, 1)

            new_feat = pointops.queryandgroup(nsample=self.nsample,
                                              xyz=xyz,
                                              new_xyz=new_xyz,
                                              feat=feat,
                                              idx=None,
                                              offset=offset,
                                              new_offset=new_offset,
                                              feature_only=True)  # (m, 4+c, nsample)
            assert new_feat.shape[-1] == self.linear.in_features
            projected = self.linear(new_feat) # sa1: (in:4+c, out:64)
            projected = projected.transpose(1, 2).contiguous()
            new_feat = self.relu(projected)  # (m, c, nsample)
            new_feat = self.pool(new_feat).squeeze(-1)  # (m, c)
            xyz, offset = new_xyz, new_offset
        else: # saa
            velocities = feat
            xyzv = torch.cat([xyz, feat], dim=1)
            assert xyzv.shape[-1] == self.linear.in_features
            projected = self.linear(xyzv) # sa1: (in:4, out:64)
            batch_normalized = self.bn(projected)
            new_feat = self.relu(batch_normalized)  # (n, c)
        return xyz, new_feat, offset, velocities