import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from .attention_block import AttentionBlock
from .weights_initialization import init_weights

"""Klasa implementująca splotową sieć neuronową "bulbul" z mechanizmem skupiania uwagi"""
class CNNAG(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # inicjalizacja warstw splotowych
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 16, kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((3, 3)))

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(16, 16, kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((3, 3)))

        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(16, 16, kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((1, 3)))

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(16, 16, kernel_size=3),
                                          torch.nn.BatchNorm2d(16),
                                          torch.nn.LeakyReLU(0.001),
                                          torch.nn.MaxPool2d((1, 3))
                                          )

        # inicjalizacja warstw w pełni połączonych

        self.flatten1_main = torch.nn.Flatten()
        self.dropout1_main = torch.nn.Dropout()
        self.fc1_main = torch.nn.Linear(7*4*16, 256)
        self.batch1_main = torch.nn.BatchNorm1d(256)
        self.leakyReLU1_main = torch.nn.LeakyReLU(0.001)

        self.dropout2_main = torch.nn.Dropout()
        self.fc2_main = torch.nn.Linear(256, 32)
        self.batch2_main = torch.nn.BatchNorm1d(32)
        self.leakyReLU2_main = torch.nn.LeakyReLU(0.001)

        self.dropout3_main = torch.nn.Dropout()

        self.fc3_main = torch.nn.Linear(32, 1)
        self.sigmoid_main = torch.nn.Sigmoid()

        self.flatten2_main = torch.nn.Flatten(start_dim=0)

        # inicjalizacja warstw w pełni połączonych
        # dla pierwszego modułu skupiania uwagi
        self.flatten1_AG1 = torch.nn.Flatten()
        self.dropout1_AG1 = torch.nn.Dropout()
        self.fc1_AG1 = torch.nn.Linear(9728, 256)
        self.batch1_AG1 = torch.nn.BatchNorm1d(256)
        self.leakyReLU1_AG1 = torch.nn.LeakyReLU(0.001)

        self.dropout2_AG1 = torch.nn.Dropout()
        self.fc2_AG1 = torch.nn.Linear(256, 32)
        self.batch2_AG1 = torch.nn.BatchNorm1d(32)
        self.leakyReLU2_AG1 = torch.nn.LeakyReLU(0.001)

        self.dropout3_AG1 = torch.nn.Dropout()

        self.fc3_AG1 = torch.nn.Linear(32, 1)
        self.sigmoid_AG1 = torch.nn.Sigmoid()

        self.flatten2_AG1 = torch.nn.Flatten(start_dim=0)

        # inicjalizacja warstw w pełni połączonych
        # dla drugiego modułu skupiania uwagi
        self.flatten1_AG2 = torch.nn.Flatten()
        self.dropout1_AG2 = torch.nn.Dropout()
        self.fc1_AG2 = torch.nn.Linear(2304, 256)
        self.batch1_AG2 = torch.nn.BatchNorm1d(256)
        self.leakyReLU1_AG2 = torch.nn.LeakyReLU(0.001)

        self.dropout2_AG2 = torch.nn.Dropout()
        self.fc2_AG2 = torch.nn.Linear(256, 32)
        self.batch2_AG2 = torch.nn.BatchNorm1d(32)
        self.leakyReLU2_AG2 = torch.nn.LeakyReLU(0.001)

        self.dropout3_AG2 = torch.nn.Dropout()

        self.fc3_AG2 = torch.nn.Linear(32, 1)
        self.sigmoid_AG2 = torch.nn.Sigmoid()

        self.flatten2_AG2 = torch.nn.Flatten(start_dim=0)

        
        # inicjalizacja modułów skupiania uwagi
        filters = [16, 16, 16, 16]
        self.attention_block1 = AttentionBlock(in_channels=filters[2],
                                                   gating_channels=filters[3],
                                                   inter_channels=filters[3],
                                                   sub_sample_factor=(1, 1),
                                                   use_phi=True, use_theta=True,
                                                   use_psi=True
                                                   )
        self.attention_block2 = AttentionBlock(in_channels=filters[3],
                                                   gating_channels=filters[3],
                                                   inter_channels=filters[3],
                                                   sub_sample_factor=(1, 1),
                                                   use_phi=True, use_theta=True,
                                                   use_psi=True
                                                   )


        # inicjalizacja agregacji
        self.aggregate = self.aggregation_mean

        # inicjalizacja wag splotowej sieci neuronowej
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                init_weights(m)
            elif isinstance(m, torch.nn.BatchNorm2d):
                init_weights(m)

        # metryki zawierające zapisaną skuteczność
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        # lista nagrań niepoprawnie zaklasyfikowanych w trakcie procesu walidacji
        self.validation_wrong_classified = []
        self.validation_wrong_classified_epoch = []


        
    """Implementacja agregacji trzech wyników uzyskanych przez sieć"""
    def aggregation_mean(self, *attended_maps):

        a1 = attended_maps[0].reshape((1, int(attended_maps[0].size()[0])))
        a2 = attended_maps[1].reshape((1, int(attended_maps[1].size()[0])))
        a3 = attended_maps[2].reshape((1, int(attended_maps[2].size()[0])))
        aggregation_cat = torch.cat((a1, a2, a3), dim=0)

        aggregation_mean = torch.mean(aggregation_cat, 0)

        return aggregation_mean

    """Implementacja działania warstw w pełni połączonych 
    dla podstawowego przebiegu sieci
    """
    def dense_layers_main(self, inputs):

        x = self.flatten1_main(inputs)
        x = self.dropout1_main(x)

        x = self.fc1_main(x)
        x = self.batch1_main(x)
        x = self.leakyReLU1_main(x)

        x = self.dropout2_main(x)
        x = self.fc2_main(x)
        x = self.batch2_main(x)
        x = self.leakyReLU2_main(x)

        x = self.dropout3_main(x)

        x = self.fc3_main(x)

        x = self.sigmoid_main(x)
        x = self.flatten2_main(x)

        return x

    """Implementacja działania warstw w pełni połączonych
    dla przebiegu sieci z pierwszym mechanizmem skupiania uwagi
    """
    def dense_layers_ag1(self, inputs):

        x = self.flatten1_AG1(inputs)
        x = self.dropout1_AG1(x)

        x = self.fc1_AG1(x)
        x = self.batch1_AG1(x)
        x = self.leakyReLU1_AG1(x)

        x = self.dropout2_AG1(x)
        x = self.fc2_AG1(x)
        x = self.batch2_AG1(x)
        x = self.leakyReLU2_AG1(x)

        x = self.dropout3_AG1(x)

        x = self.fc3_AG1(x)

        x = self.sigmoid_AG1(x)
        x = self.flatten2_AG1(x)

        return x

    """Implementacja działania warstw w pełni połączonych
    dla przebiegu sieci z drugim mechanizmem skupiania uwagi
    """
    def dense_layers_ag2(self, inputs):

        x = self.flatten1_AG2(inputs)
        x = self.dropout1_AG2(x)

        x = self.fc1_AG2(x)
        x = self.batch1_AG2(x)
        x = self.leakyReLU1_AG2(x)

        x = self.dropout2_AG2(x)
        x = self.fc2_AG2(x)
        x = self.batch2_AG2(x)
        x = self.leakyReLU2_AG2(x)

        x = self.dropout3_AG2(x)

        x = self.fc3_AG2(x)

        x = self.sigmoid_AG2(x)
        x = self.flatten2_AG2(x)

        return x


    """Główna metoda implementująca proces działania sieci"""
    def forward(self, inputs):

        # warstwy splotowe
        conv_layer1 = self.layer1(inputs)
        conv_layer2 = self.layer2(conv_layer1)
        conv_layer3 = self.layer3(conv_layer2)
        conv_layer4 = self.layer4(conv_layer3)

        # warstwa w pełni połączona
        after_dense = self.dense_layers_main(conv_layer4)

        # moduł skupiania uwagi
        g_conv1, att1 = self.attention_block1(conv_layer2, conv_layer4)
        g_conv2, att2 = self.attention_block2(conv_layer3, conv_layer4)

        # warstwy w pełni połączone po mechanizmie skupiania uwagi
        g1 = self.dense_layers_ag1(g_conv1)
        g2 = self.dense_layers_ag2(g_conv2)

        output = self.aggregate(g1, g2, after_dense)
        return output

    """Funkcja straty - binarna entropia krzyżowa"""
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

        # utworzenie listy z nazwami plików źle zaklasyfikowanych
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

    """Implementacja optymizatora ADAM 
        wraz z redukcją paramtru uczenia się, 
        jeśli po pięciu epokach nie było poprawy
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'validation_epoch_end_loss'
        }
