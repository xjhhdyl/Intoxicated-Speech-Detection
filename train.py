import yaml
import torch
from tqdm import tqdm
from model.classification_model import ECABasicBlock, interattention, rescrossSE, mmWavoiceNet, rnnClassfication, \
    mmWavoice
from solver.solver import mmWavoice_batch_iterator
from utils.data import mmWavoiceDataset, mmWavoiceLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

if __name__ == '__main__':
    writer = SummaryWriter("./logs_train")
    config_path = "./config/Wavoice-config.yaml"

    # Fix seed
    seed = 3047
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set cuda device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("------------------------------------------------")
    print("Loading Config", flush=True)
    # load config file
    print("Loading configure at", config_path)
    with open(config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    data_name = params["data"]["name"]
    print(data_name)

    tf_rate_upperbound = params["training"][
        "tf_rate_upperbound"]  ## teacher forcing rate during training will be linearly in las
    tf_rate_lowerbound = params["training"][
        "tf_rate_lowerbound"]  # decaying from upperbound to lower bound for each epoch in las
    tf_decay_step = params["training"]["tf_decay_step"]
    epochs = params["training"]["epochs"]

    # Load datasets
    print("------------------------------------------------")
    print("Processing datasets...", flush=True)
    train_dataset = mmWavoiceDataset(params, "train")  # AudioDataset(params, "train")
    train_loader = mmWavoiceLoader(train_dataset, shuffle=True, num_workers=params["data"]["num_works"]).loader
    dev_dataset = mmWavoiceDataset(params, "test")  # AudioDataset(params,"test")
    dev_loader = mmWavoiceLoader(dev_dataset, num_workers=params["data"]["num_works"]).loader

    print("------------------------------------------------")
    print("Creating model architecture...", flush=True)

    mmwavoicenet = mmWavoiceNet(ECABasicBlock, rescrossSE, interattention, [1, 1, 1, 1, 1])
    rnnclassfication = rnnClassfication(40, 128, 2)
    mmWavoice_model = mmWavoice(rnnclassfication, mmwavoicenet)

    mmWavoice_model.cuda()

    # Create optimizer
    optimizer = torch.optim.Adam(params=mmWavoice_model.parameters(), lr=params["training"]["lr"])

    print("------------------------------------------------")
    print("Training...", flush=True)

    global_step = 0 + len(train_loader)
    my_fields = {"loss": 0}
    best_acc = 0.0
    for epoch in tqdm(range(epochs), desc="Epoch training"):
        epoch_step = 0
        train_epoch_acc = 0
        train_loss = []
        batch_loss = 0
        train_corrects = 0
        for i, (data) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False,
                              desc=f"Epoch number {epoch}"):
            my_fields["loss"] = batch_loss
            tf_rate = tf_rate_upperbound - (tf_rate_upperbound - tf_rate_lowerbound) * min(
                (float(global_step) / tf_decay_step), 1
            )  # adjust learning

            voice_inputs = data[1]["voice_inputs"].cuda()
            mmwave_inputs = data[2]["mmwave_inputs"].cuda()
            labels = data[3]["targets"].cuda()

            batch_loss, batch_corrects = mmWavoice_batch_iterator(
                voice_batch_data=voice_inputs,
                mmwave_batch_data=mmwave_inputs,
                batch_label=labels,
                mmWavoice_model=mmWavoice_model,
                optimizer=optimizer,
                is_training=True,
            )
            torch.cuda.empty_cache()
            train_corrects += batch_corrects
            train_loss.append(batch_loss)
            global_step += 1
            epoch_step += 1
            writer.add_scalar("loss/train_step", batch_loss, global_step)
        train_loss = np.array([sum(train_loss) / len(train_loss)])
        train_epoch_acc = train_corrects / len(train_loader.dataset)
        writer.add_scalar("loss/train-epoch", train_loss, epoch)
        writer.add_scalar("acc/train-epoch", train_epoch_acc, epoch)
        print(f"training_epoch:{epoch},train_loss:{train_loss},traning_epoch_acc:{train_epoch_acc}")
        # valiation
        val_loss = []
        val_acc = 0
        val_step = 0
        dev_correncts = 0
        for i, (data) in tqdm(enumerate(dev_loader), total=len(dev_loader), leave=False, desc="Validation"):
            with torch.no_grad():
                voice_inputs = data[1]["voice_inputs"].cuda()
                mmwave_inputs = data[2]["mmwave_inputs"].cuda()
                labels = data[3]["targets"].cuda()

                batch_loss, batch_corrects = mmWavoice_batch_iterator(
                    voice_batch_data=voice_inputs,
                    mmwave_batch_data=mmwave_inputs,
                    batch_label=labels,
                    mmWavoice_model=mmWavoice_model,
                    optimizer=optimizer,
                    is_training=False,
                )
                torch.cuda.empty_cache()
                dev_correncts += batch_corrects
                val_loss.append(batch_loss)
                val_step += 1

        val_loss = np.array([sum(val_loss) / len(val_loss)])
        val_acc = dev_correncts / len(dev_loader.dataset)
        writer.add_scalar("loss/dev", val_loss, epoch)
        writer.add_scalar("acc/dev", val_acc, epoch)
        print(f"dev_epoch:{epoch},val_loss:{val_loss},dev_epoch_acc:{val_acc}")
        # deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            file_path = r"./model.pth"
            torch.save(mmWavoice_model.state_dict(), file_path)
