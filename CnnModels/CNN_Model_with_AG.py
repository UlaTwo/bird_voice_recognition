import torch
from torch.nn import functional as F
import pytorch_lightning as pl

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from CnnModels.weights_initialization import init_weights
from CnnModels.GridAttentionBlock import GridAttentionBlock

class CNN_Model_with_AG(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        filters = [16,16,16,16]
        
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        #convolution layers
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1,16,kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((3,3)) )
        
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(16,16,kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((3,3)) )
        
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(16,16,kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((1,3)))

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(16,16,kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((1,3))
                                         )
        
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        #dense layers
        
        self.flatten1_main = torch.nn.Flatten()
        self.dropout1_main = torch.nn.Dropout()
        self.fc1_main = torch.nn.Linear(7*4*16,256)
        self.batch1_main = torch.nn.BatchNorm1d(256) 
        self.leakyReLU1_main = torch.nn.LeakyReLU(0.001)
        
        self.dropout2_main = torch.nn.Dropout()
        self.fc2_main = torch.nn.Linear(256,32)
        self.batch2_main = torch.nn.BatchNorm1d(32) 
        self.leakyReLU2_main = torch.nn.LeakyReLU(0.001)
        
        self.dropout3_main = torch.nn.Dropout()
        
        
        self.fc3_main = torch.nn.Linear(32,1)
        self.sigmoid_main = torch.nn.Sigmoid()
        
        self.flatten2_main = torch.nn.Flatten(start_dim=0)
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        #dense layers for first AG
        self.flatten1_AG1 = torch.nn.Flatten()
        self.dropout1_AG1 = torch.nn.Dropout()
        self.fc1_AG1 = torch.nn.Linear(9728,256)
        self.batch1_AG1 = torch.nn.BatchNorm1d(256) 
        self.leakyReLU1_AG1 = torch.nn.LeakyReLU(0.001)
        
        self.dropout2_AG1 = torch.nn.Dropout()
        self.fc2_AG1 = torch.nn.Linear(256,32)
        self.batch2_AG1 = torch.nn.BatchNorm1d(32)
        self.leakyReLU2_AG1 = torch.nn.LeakyReLU(0.001)
        
        self.dropout3_AG1 = torch.nn.Dropout()
        
        
        self.fc3_AG1 = torch.nn.Linear(32,1)
        self.sigmoid_AG1 = torch.nn.Sigmoid()
        
        self.flatten2_AG1 = torch.nn.Flatten(start_dim=0)
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        #dense layers for second AG
        self.flatten1_AG2 = torch.nn.Flatten()
        self.dropout1_AG2 = torch.nn.Dropout()
        self.fc1_AG2 = torch.nn.Linear(2304,256)
        self.batch1_AG2 = torch.nn.BatchNorm1d(256) 
        self.leakyReLU1_AG2 = torch.nn.LeakyReLU(0.001)
        
        self.dropout2_AG2 = torch.nn.Dropout()
        self.fc2_AG2 = torch.nn.Linear(256,32)
        self.batch2_AG2 = torch.nn.BatchNorm1d(32)
        self.leakyReLU2_AG2 = torch.nn.LeakyReLU(0.001)
        
        self.dropout3_AG2 = torch.nn.Dropout()
        
        
        self.fc3_AG2 = torch.nn.Linear(32,1)
        self.sigmoid_AG2 = torch.nn.Sigmoid()
        
        self.flatten2_AG2 = torch.nn.Flatten(start_dim=0)
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        #attention gates
        self.attention_block1 = GridAttentionBlock(in_channels = filters[2], 
                                                     gating_channels = filters[3],
                                                     inter_channels = filters[3],
                                                     sub_sample_factor = (1,1),
                                                     use_phi=True, use_theta = True,
                                                     use_psi = True
                                                    )
        self.attention_block2 = GridAttentionBlock(in_channels = filters[3], 
                                                     gating_channels = filters[3],
                                                     inter_channels = filters[3],
                                                     sub_sample_factor = (1,1),
                                                     use_phi=True,use_theta = True,
                                                     use_psi = True
                                                    )
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        # Aggragation  
        self.aggregate = self.aggregation_mean
        
        # # # # # # # # # # # # # # # # # # # # # # # #
        # initialise weights

        # The self.modules() method returns an iterable to the many layers or “modules” defined in the model class.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init_weights(m, init_type = 'kaiming')
            elif isinstance(m, torch.nn.BatchNorm2d):
                init_weights(m, init_type = 'kaiming')
        
        
        # compute the accuracy
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        
        # list of files wrong classified during validation epoch
        self.validation_wrong_classified = []
        self.validation_wrong_classified_epoch = []
        
    # ***************************************************** #
    #  Arguments: 
    #     attended_maps - (g1,g2,g)
    def aggregation_mean(self, *attended_maps):

        a1 = attended_maps[0].reshape( (1,int( attended_maps[0].size()[0] ) ) )
        a2 = attended_maps[1].reshape((1,int( attended_maps[1].size()[0]) )) 
        a3 = attended_maps[2].reshape((1,int( attended_maps[2].size()[0]) ))
        aggregation_cat = torch.cat((a1,a2,a3), dim=0)

        aggregation_mean = torch.mean(aggregation_cat, 0)

        return aggregation_mean
    
    
    # ***************************************************** #
    def dense_layers_main(self,inputs):

        x=self.flatten1_main(inputs)
        x=self.dropout1_main(x)
        
        x=self.fc1_main(x)
        x=self.batch1_main(x)
        x=self.leakyReLU1_main(x)
        
        x=self.dropout2_main(x)
        x=self.fc2_main(x)
        x=self.batch2_main(x)
        x=self.leakyReLU2_main(x)
        
        x=self.dropout3_main(x)
        
        x=self.fc3_main(x)
        
        x = self.sigmoid_main(x)
        x=self.flatten2_main(x)
        
        return x
        
    # ***************************************************** #
    def dense_layers_AG1(self,inputs):

        x=self.flatten1_AG1(inputs)
        x=self.dropout1_AG1(x)
        
        x=self.fc1_AG1(x)
        x=self.batch1_AG1(x)
        x=self.leakyReLU1_AG1(x)
        
        x=self.dropout2_AG1(x)
        x=self.fc2_AG1(x)
        x=self.batch2_AG1(x)
        x=self.leakyReLU2_AG1(x)
        
        x=self.dropout3_AG1(x)
        
        x=self.fc3_AG1(x)
        
        x = self.sigmoid_AG1(x)
        x=self.flatten2_AG1(x)
        
        return x
        
    # ***************************************************** #
    def dense_layers_AG2(self,inputs):

        x=self.flatten1_AG2(inputs)
        x=self.dropout1_AG2(x)
        
        x=self.fc1_AG2(x)
        x=self.batch1_AG2(x)
        x=self.leakyReLU1_AG2(x)
        
        x=self.dropout2_AG2(x)
        x=self.fc2_AG2(x)
        x=self.batch2_AG2(x)
        x=self.leakyReLU2_AG2(x)
        
        x=self.dropout3_AG2(x)
        
        x=self.fc3_AG2(x)
        
        x = self.sigmoid_AG2(x)
        x=self.flatten2_AG2(x)
        
        return x
        
    # ***************************************************** #
        
    def forward(self,inputs):
        
        #convolution layers
        conv_layer1=self.layer1(inputs)
        conv_layer2=self.layer2(conv_layer1)
        conv_layer3=self.layer3(conv_layer2)
        conv_layer4=self.layer4(conv_layer3)
        
        after_dense = self.dense_layers_main(conv_layer4)
        
        ## Attention Mechanism
        g_conv1, att1 = self.attention_block1(conv_layer2 ,conv_layer4)
        g_conv2, att2 = self.attention_block2(conv_layer3,conv_layer4)
        
        # dense_layers after attention_block
        g1 = self.dense_layers_AG1(g_conv1)
        g2 = self.dense_layers_AG2(g_conv2)
        
        output = self.aggregate(g1,g2,after_dense)
        return output
    
    # ***************************************************** #
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
        
        list_file_names = []
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
