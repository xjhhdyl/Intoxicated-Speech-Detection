import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

#  可添加label_smoothing技术
#  样本不平衡
def mmWavoice_batch_iterator(
        voice_batch_data,
        mmwave_batch_data,
        batch_label,
        mmWavoice_model,
        optimizer,
        is_training,
):
    criterion = WeightedFocalLoss().cuda()
    optimizer.zero_grad()
    outputs = mmWavoice_model(
        voice_batch_data=voice_batch_data, mmwave_batch_data=mmwave_batch_data, batch_label=batch_label,
        is_training=is_training,
    )
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, batch_label.long())
    batch_loss = loss.cpu().data.numpy()

    if is_training:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mmWavoice_model.parameters(), 1)
        optimizer.step()
    batch_corrects = torch.sum(preds == batch_label.data)
    return batch_loss, batch_corrects
