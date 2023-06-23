import torch
import torch.nn as nn

b = 0.2


#  样本不平衡
def mmWavoice_batch_iterator(
        voice_batch_data,
        mmwave_batch_data,
        batch_label,
        mmWavoice_model,
        optimizer,
        is_training,
):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer.zero_grad()
    outputs = mmWavoice_model(
        voice_batch_data=voice_batch_data, mmwave_batch_data=mmwave_batch_data, batch_label=batch_label,
        is_training=is_training,
    )
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, batch_label)
    # flood = (loss - b).abs() + b
    batch_loss = loss.cpu().data.numpy()

    if is_training:
        loss.backward()
        # flood.backward()
        torch.nn.utils.clip_grad_norm_(mmWavoice_model.parameters(), 1)
        optimizer.step()
    batch_corrects = torch.sum(preds == batch_label.data)
    return batch_loss, batch_corrects
