import torch
import torch.nn as nn




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def compute_contrastive_weight_mat(self, weight_net, features, batch_size):
        # weight_mat = torch.zeros([features.shape[0], features.shape[0]], device = features.device)

        transformed_features = weight_net(features)

        feature_norm = torch.norm(transformed_features, dim=1)

        # print(torch.min(feature_norm))

        # transformed_features = transformed_features/(feature_norm.view(-1,1) + 1e-6)

        weight_mat = torch.matmul(transformed_features, transformed_features.T)

        weight_mat = torch.sigmoid(weight_mat)
        # feature_norm_prod = torch.mm(feature_norm.view(-1,1), feature_norm.view(1,-1))

        # weight_mat = weight_mat/feature_norm_prod

        # for idx1 in range(0,features.shape[0], batch_size):
        #     end_idx1 = idx1 + batch_size
        #     if end_idx1 > features.shape[0]:
        #         end_idx1 = features.shape[0]
        #     for idx2 in range(0, features.shape[0], batch_size):
        #         end_idx2 = idx2 + batch_size
        #         if end_idx2 > features.shape[0]:
        #             end_idx2 = features.shape[0]

        #         feature_ls1 = features[idx1:end_idx1]
        #         feature_ls2 = features[idx2:end_idx2]
        #         all_feature = torch.cat([feature_ls1, feature_ls2], dim=1)

        #         weight_mat[idx1:end_idx1, idx2:end_idx2] = weight_net(all_feature)
        del transformed_features, feature_norm
        return weight_mat

    def transform_weight_mat(self, weight_mat):
        top2_items = torch.topk(weight_mat, k=2, dim=-1)[0]
        half_gap = top2_items[:,0] - top2_items[:,1] + 1e-6
        top2_means = (top2_items[:,0] + top2_items[:,1])/2
        transformed_weight_mat = torch.relu(weight_mat - top2_means.unsqueeze(1))/half_gap.unsqueeze(1)*2 * top2_items[:,0].unsqueeze(1)
        return transformed_weight_mat

    def forward(self, features, labels=None, mask=None, weight_mat = None, bias_mat = None, pos_weight_net = None, neg_weight_net = None, mb_size = 128):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if weight_mat is not None:
            weight_mat = torch.softmax(weight_mat, dim=-1)
            weight_mat = self.transform_weight_mat(weight_mat)
            # weight_mat = torch.sum(weight_mat.unsqueeze(0)*weight_mat.unsqueeze(1), dim=-1)
            weight_mat = torch.max(weight_mat.unsqueeze(0)*weight_mat.unsqueeze(1), dim=-1)[0]

        if labels is not None:
            if len(torch.unique(labels)) <= 1:
                return 0
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # pos_weight_mat = None
        # neg_weight_mat = None
        # if weight_mat is not None:
        #     pos_weight_mat = weight_mat
        #     neg_weight_mat = weight_mat
        # else:
        #     if pos_weight_net is not None:
        #         pos_weight_mat = self.compute_contrastive_weight_mat(pos_weight_net, contrast_feature, mb_size)
        #     if neg_weight_net is not None:
                
        #         neg_weight_mat = self.compute_contrastive_weight_mat(neg_weight_net, contrast_feature, mb_size)
                # print('min neg weight::', torch.min(neg_weight_mat))
                # print('max neg weight::', torch.max(neg_weight_mat))
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_feature_norm = anchor_feature.norm(dim=-1) + 1e-5
        logits = anchor_dot_contrast/torch.mm(anchor_feature_norm.view(-1,1), anchor_feature_norm.view(1,-1))
        # if weight_mat is not None:
        #     logits *= weight_mat
        # if bias_mat is not None:
        #     logits += bias_mat
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        inverse_mask = ~(mask.bool())
        inverse_mask = inverse_mask.int()*logits_mask
        # inverse_mask = torch.ones_like(mask)

        mask = mask * logits_mask

        # compute log_prob
        # if neg_weight_mat is None:
        # exp_logits = torch.exp(logits) * logits_mask*inverse_mask
        # else:
        if weight_mat is not None:
            exp_logits = torch.exp(logits) * logits_mask*(1-weight_mat)
        else:
            exp_logits = torch.exp(logits) * logits_mask*inverse_mask
            # exp_logits = exp_logits*neg_weight_mat


        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # if pos_weight_mat is not None:
        #     log_prob += pos_weight_mat

        # compute mean of log-likelihood over positive
        if weight_mat is not None:
            masked_log_prob = logits_mask * log_prob * weight_mat
        else:
            masked_log_prob = mask * log_prob
        # if pos_weight_mat is not None:
        #     masked_log_prob = masked_log_prob+pos_weight_mat
        if weight_mat is not None:
            mean_log_prob_pos = (masked_log_prob).mean(1)
        else:
            mean_log_prob_pos = (masked_log_prob).sum(1) / (mask.sum(1) + 1e-5)
        # mean_log_prob_pos[mask.sum(1)==0] = 0

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print("loss::", loss)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return loss


    def forward2(self, features, labels=None, mask=None, weight_mat = None, bias_mat = None, pos_weight_net = None, neg_weight_net = None, mb_size = 128):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if labels is not None:
            if len(torch.unique(labels)) <= 1:
                return 0
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # pos_weight_mat = None
        # neg_weight_mat = None
        # if weight_mat is not None:
        #     pos_weight_mat = weight_mat
        #     neg_weight_mat = weight_mat
        # else:
        #     if pos_weight_net is not None:
        #         pos_weight_mat = self.compute_contrastive_weight_mat(pos_weight_net, contrast_feature, mb_size)
        #     if neg_weight_net is not None:
                
        #         neg_weight_mat = self.compute_contrastive_weight_mat(neg_weight_net, contrast_feature, mb_size)
                # print('min neg weight::', torch.min(neg_weight_mat))
                # print('max neg weight::', torch.max(neg_weight_mat))
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        anchor_feature_norm = anchor_feature.norm(dim=-1) + 1e-5
        logits = anchor_dot_contrast/torch.mm(anchor_feature_norm.view(-1,1), anchor_feature_norm.view(1,-1))
        if weight_mat is not None:
            logits *= weight_mat
        if bias_mat is not None:
            logits += bias_mat
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        inverse_mask = ~(mask.bool())
        inverse_mask = inverse_mask.int()*logits_mask
        # inverse_mask = torch.ones_like(mask)

        mask = mask * logits_mask

        # compute log_prob
        # if neg_weight_mat is None:
        exp_logits = torch.exp(logits) * logits_mask*inverse_mask
        # else:
        #     exp_logits = torch.exp(logits+neg_weight_mat) * logits_mask*inverse_mask
            # exp_logits = exp_logits*neg_weight_mat


        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # if pos_weight_mat is not None:
        #     log_prob += pos_weight_mat

        # compute mean of log-likelihood over positive
        masked_log_prob = mask * log_prob
        # if pos_weight_mat is not None:
        #     masked_log_prob = masked_log_prob+pos_weight_mat
        mean_log_prob_pos = (masked_log_prob).sum(1) / (mask.sum(1) + 1e-5)
        # mean_log_prob_pos[mask.sum(1)==0] = 0

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        # print("loss::", loss)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


        return loss
