import torch
from torch.nn import functional as F
import pytorch_lightning as pl


class CNN_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #convolution layers
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1,16,kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((3,3)) )
        
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(16,16,kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((3,3)) )
        
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(16,16,(3,3)),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((1,3)))

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(16,16,kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((1,3)),
                                          torch.nn.Flatten())
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        #dense layers
        
        self.dropout1 = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(7*4*16,256)
        self.batch1 = torch.nn.BatchNorm1d(256) 
        self.leakyReLU1 = torch.nn.LeakyReLU(0.001)
        
        self.dropout2 = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(256,32)
        self.batch2 = torch.nn.BatchNorm1d(32) 
        self.leakyReLU2 = torch.nn.LeakyReLU(0.001)
        
        self.dropout3 = torch.nn.Dropout()
        
        
        self.fc3 = torch.nn.Linear(32,1) 
        self.sigmoid = torch.nn.Sigmoid()
        
        self.flatten = torch.nn.Flatten(start_dim=0)
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        
        # compute the accuracy
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
        # list of files wrong classified during validation epoch
        self.validation_wrong_classified = []
        self.validation_wrong_classified_epoch = []
        
    def forward(self,x):
        print("początek forward")
        #convolution layers
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x) 

        #dense layers
        x=self.dropout1(x)
        x=self.fc1(x)
        x=self.batch1(x)
        x=self.leakyReLU1(x)
        
        x=self.dropout2(x)
        x=self.fc2(x)
        x=self.batch2(x)
        x=self.leakyReLU2(x)
        
        x=self.dropout3(x)
        
        x=self.fc3(x)
        
        x = self.sigmoid(x)
        
        x=self.flatten(x)

        print("koniec forward")
        return x
    
    def cross_entropy_loss(self, logits, labels):
        return F.binary_cross_entropy(logits, labels)

    def training_step(self, train_batch, batch_idx):

        x, y, f = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        
        y = y.int()
        accuracy = self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_epoch=True, sync_dist=True)

        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, val_batch, batch_idx):

        x, y, f = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        y = y.int()
        accuracy = self.valid_acc(logits, y)
        
        # creating list of files wrong classified during validation
        for id in range(len(f)):
            if round(float(logits[id])) != y[id]:
                self.validation_wrong_classified_epoch.append(f[id])
                

        return {'val_loss': loss, 'val_accuracy': accuracy}

    def test_step(self, test_batch, batch_idx):
        x, y, f = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        y = y.int()
        accuracy = self.test_acc(logits, y)

        return {'test_loss': loss, 'test_accuracy': accuracy}
    
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()

        self.log('training_epoch_end_accuracy', avg_accuracy, sync_dist=True)
        self.log('training_epoch_end_loss', avg_loss, sync_dist=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], sync_dist=True)
    
    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        
        self.log('validation_epoch_end_accuracy', avg_accuracy, sync_dist=True)
        self.log('validation_epoch_end_loss', avg_loss, sync_dist=True)
        
        self.validation_wrong_classified.append(self.validation_wrong_classified_epoch.copy())
        self.validation_wrong_classified_epoch.clear()

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        
        self.log('test_epoch_end_accuracy', avg_accuracy, sync_dist=True)
        self.log('test_epoch_end_loss', avg_loss, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'validation_epoch_end_loss'
        }
