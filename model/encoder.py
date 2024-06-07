import torch.nn as nn

class SpectralEncoder(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpectralEncoder, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        self.conv1 = nn.Conv3d(1, self.inter_size, kernel_size=(7, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                               bias=True)
        self.bn1 = nn.BatchNorm3d(self.inter_size)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn2 = nn.BatchNorm3d(self.inter_size)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), padding_mode='zeros', bias=True)
        self.bn3 = nn.BatchNorm3d(self.inter_size)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv3d(self.inter_size, self.feature_dim,
                               kernel_size=(((self.input_channels - 7 + 2 * 1) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(self.feature_dim)
        self.activation4 = nn.ReLU()

        self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))

        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))

        x1 = self.avgpool(x1)
        x1 = x1.reshape((x1.size(0), -1))

        return x1


class SpatialEncoder(nn.Module):
    def __init__(self, input_channels, patch_size, feature_dim):
        super(SpatialEncoder, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.inter_size = 24

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, self.inter_size, kernel_size=(self.input_channels, 1, 1))
        self.bn5 = nn.BatchNorm3d(self.inter_size)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv8 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 1, 1))

        self.conv6 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn6 = nn.BatchNorm3d(self.inter_size)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode='zeros', bias=True)
        self.bn7 = nn.BatchNorm3d(self.inter_size)
        self.activation7 = nn.ReLU()

        self.avgpool = nn.AvgPool3d((1, self.patch_size, self.patch_size))

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(self.inter_size, out_features=self.feature_dim))


    def forward(self, x):
        x = x.unsqueeze(1)

        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv6(x2)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))

        x2 = self.avgpool(x2)
        x2 = x2.reshape((x2.size(0), -1))

        x2 = self.fc(x2)

        return x2


class WordEmbTransformers(nn.Module):
    def __init__(self, feature_dim, dropout):
        super(WordEmbTransformers, self).__init__()
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.fc = nn.Sequential(nn.Linear(in_features=768,
                                          out_features=128,
                                          bias=True),
                                nn.ReLU(),
                                nn.Dropout(p=self.dropout),
                                nn.Linear(in_features=128,
                                          out_features=self.feature_dim,
                                          bias=True)
                                )

    def forward(self, x):
        # 0-1
        x = self.fc(x)
        return x


class AttentionWeight(nn.Module):
    def __init__(self, feature_dim, hidden_layer, dropout):
        super(AttentionWeight, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_layer = hidden_layer
        self.dropout = dropout

        self.getAttentionWeight = nn.Sequential(nn.Linear(in_features=self.feature_dim,
                                                          out_features=self.hidden_layer),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(in_features=self.hidden_layer,
                                                          out_features=1),
                                                nn.Sigmoid()
                                                )

    def forward(self, x):
        x = self.getAttentionWeight(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_dimension, patch_size, emb_size, dropout=0.5):
        super(Encoder, self).__init__()
        self.n_dimension = n_dimension
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.dropout = dropout

        self.spectral_encoder = SpectralEncoder(input_channels=self.n_dimension, patch_size=self.patch_size, feature_dim=self.emb_size)
        self.spatial_encoder = SpatialEncoder(input_channels=self.n_dimension, patch_size=self.patch_size, feature_dim=self.emb_size)
        self.word_emb_transformers = WordEmbTransformers(feature_dim=self.emb_size, dropout=self.dropout)

    def forward(self, x, semantic_feature="", s_or_q="query"):
        spatial_feature = self.spatial_encoder(x)
        spectral_feature = self.spectral_encoder(x)
        spatial_spectral_fusion_feature = 0.5 * spatial_feature + 0.5 * spectral_feature

        # support set extract fusion_feature
        if s_or_q == "support":  # semantic_feature = (9, 768)
            semantic_feature = self.word_emb_transformers(semantic_feature)  # (9, 128)
            return spatial_spectral_fusion_feature, semantic_feature
        # query set extract spatial_spectral_fusion_feature
        return spatial_spectral_fusion_feature