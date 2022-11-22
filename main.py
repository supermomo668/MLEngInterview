from instadeep.dataset import SequenceDataset
from instadeep.model import ProtCNN
from instadeep.utils import
#
import argparse

pl.seed_everything(0)

class ProtCNN(pl.LightningModule):
    
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(22, 128, kernel_size=1, padding=0, bias=False),
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            torch.nn.Linear(7680, num_classes)
        )
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        
    def forward(self, x):
        return self.model(x.float())
    
    def training_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)        
        acc = self.valid_acc(pred, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=1e-2, momentum=0.9, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                            milestones=[5, 8, 10, 12, 14, 16, 18, 20], gamma=0.9)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    

def main(args):
    #
    trainer = pl.Trainer(gpus=gpus, max_epochs=epochs)
    if args.mode=="train":
        train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "train")
        dev_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "dev")
        dataloaders['train'] = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
        )
        dataloaders['dev'] = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
        )
        trainer.fit(prot_cnn, dataloaders['train'], dataloaders['dev'])
        
    elif args.mode=="test:
        test_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "test")
        dataloaders['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
        )
        model = MyLightningModule.load_from_checkpoint(
            checkpoint_path="/path/to/pytorch_checkpoint.ckpt",
            hparams_file="/path/to/experiment/version/hparams.yaml",
            map_location=None,
        )

        # init trainer with whatever options
        trainer = Trainer(...)

        # test (pass in the model)
        trainer.test(model)
    
    
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', '-data_dir', help='path to data', type=str,
                    default='./random_split')
    ap.add_argument('--mode', '-mode', required=True, type=str,
                    choices  = ["train","test"])
    args = ap.parse_args()
    
    # run main
    main(args)