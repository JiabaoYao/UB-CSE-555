import copy
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder
import numpy as np
from typing import Optional, Union, Callable
from siformer.attention import AttentionLayer, ProbAttention, FullAttention
from siformer.decoder import DecoderLayer, PBEEDecoder
from siformer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, PBEEncoder
from siformer.utils import get_sequence_list

import uuid


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class FeatureIsolatedTransformer(nn.Transformer):
    def __init__(self, d_model_list: list, nhead_list: list, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 selected_attn: str = 'prob', output_attention: str = True,
                 inner_classifiers_config: list = None, patience: int = 1, use_pyramid_encoder: bool = False,
                 distil: bool = False, projections_config: list = None,
                 IA_encoder: bool = False, IA_decoder: bool = False, device=None):

        super(FeatureIsolatedTransformer, self).__init__(sum(d_model_list), nhead_list[-1], num_encoder_layers,
                                                         num_decoder_layers, dim_feedforward, dropout, activation)
        del self.encoder
        self.d_model = sum(d_model_list)
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.device = device
        self.use_pyramid_encoder = use_pyramid_encoder
        self.use_IA_encoder = IA_encoder
        self.use_IA_decoder = IA_decoder
        self.inner_classifiers_config = inner_classifiers_config
        self.projections_config = projections_config
        self.patience = patience
        self.distil = distil
        self.activation = activation
        self.selected_attn = selected_attn
        self.output_attention = output_attention
        self.l_hand_encoder = self.get_custom_encoder(d_model_list[0], nhead_list[0])
        self.r_hand_encoder = self.get_custom_encoder(d_model_list[1], nhead_list[1])
        self.body_encoder = self.get_custom_encoder(d_model_list[2], nhead_list[2])
        self.decoder = self.get_custom_decoder(nhead_list[-1])
        self._reset_parameters()

    def get_custom_encoder(self, f_d_model: int, nhead: int):
        Attn = ProbAttention if self.selected_attn == 'prob' else FullAttention
        print(f'self.selected_attn {self.selected_attn}')

        if self.use_pyramid_encoder:
            print("Pyramid encoder")
            print(f'self.distl {self.distil}')
            e_layers = get_sequence_list(self.num_encoder_layers)
            inp_lens = list(range(len(e_layers)))
            encoders = [
                Encoder(
                    [
                        EncoderLayer(
                            AttentionLayer(
                                Attn(output_attention=self.output_attention),
                                f_d_model, nhead, mix=False),
                            f_d_model,
                            self.d_ff,
                            dropout=self.dropout,
                            activation=self.activation
                        ) for _ in range(el)
                    ],
                    [
                        ConvLayer(
                            f_d_model, self.device
                        ) for _ in range(self.num_encoder_layers - 1)
                    ] if self.distil else None,
                    norm_layer=torch.nn.LayerNorm(f_d_model)
                ) for el in e_layers]

            encoder = EncoderStack(encoders, inp_lens)
        else:
            encoder_layer = TransformerEncoderLayer(f_d_model, nhead, self.d_ff, self.dropout, self.activation)
            encoder_layer.self_attn = AttentionLayer(
                Attn(output_attention=self.output_attention),
                f_d_model, nhead, mix=False
            )
            encoder_norm = LayerNorm(f_d_model)

            if self.use_IA_encoder:
                print("Encoder with input adaptive")
                self.inner_classifiers_config[0] = f_d_model
                encoder = PBEEncoder(
                    encoder_layer, self.num_encoder_layers, norm=encoder_norm,
                    inner_classifiers_config=self.inner_classifiers_config,
                    projections_config=self.projections_config,
                    patience=self.patience
                )
            else:
                print("Normal encoder")
                encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, norm=encoder_norm)

        return encoder

    def get_custom_decoder(self, nhead):
        decoder_layer = DecoderLayer(self.d_model, nhead, self.d_ff)
        decoder_norm = LayerNorm(self.d_model)
        if self.use_IA_decoder:
            print("Decoder with with input adaptive")
            return PBEEDecoder(
                decoder_layer, self.num_decoder_layers, norm=decoder_norm,
                 patient=self.patience,inner_classifiers_config=self.inner_classifiers_config
            )
        else:
            print("Normal decoder")
            return TransformerDecoder(
                decoder_layer, self.num_decoder_layers, norm=decoder_norm)

    def checker(self, full_src, tgt, is_batched):
        if not self.batch_first and full_src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and full_src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if full_src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

    def forward(self, src: list, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                src_is_causal: Optional[bool] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False, training: bool = True) -> Tensor:

        full_src = torch.cat(src, dim=-1)
        self.checker(full_src, tgt, full_src.dim() == 3)

        id = uuid.uuid1()
        # code for concurrency is removed...
        if self.use_IA_encoder:
            l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask, training=training)
            r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask, training=training)
            body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask, training=training)
        else:
            l_hand_memory = self.l_hand_encoder(src[0], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            r_hand_memory = self.r_hand_encoder(src[1], mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            body_memory = self.body_encoder(src[2], mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        full_memory = torch.cat((l_hand_memory, r_hand_memory, body_memory), -1)

        if self.use_IA_decoder:
            output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask, training=training)
        else:
            output = self.decoder(tgt, full_memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        return output


class AbsolutePE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(AbsolutePE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)
        # reshape the matrix shape to met the input shape
        # pe = pe.squeeze(0).unsqueeze(1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class SpoTer(nn.Module):
    def __init__(self, num_classes, num_hid=108, num_enc_layers=3, num_dec_layers=2, seq_len=204):
        super(SpoTer, self).__init__()
        print("Normal transformer")
        self.embedding = AbsolutePE(d_model=num_hid, max_len=seq_len)
        self.class_query = nn.Parameter(torch.rand(1, num_hid))
        self.transformer = nn.Transformer(num_hid, 9, num_enc_layers, num_dec_layers)
        custom_decoder_layer = DecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048, 0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)
        self.projection = nn.Linear(num_hid, num_classes)
        print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}")

    def forward(self, l_hand, r_hand, body, training):
        batch_size = l_hand.size(0)

        inputs = torch.cat((l_hand, r_hand, body), -2)
        inputs = inputs.view(inputs.size(0), inputs.size(1), inputs.size(2) * inputs.size(3))

        new_inputs = self.embedding(inputs)
        new_inputs = new_inputs.permute(1, 0, 2).type(dtype=torch.float32)

        transformer_out = self.transformer(new_inputs, self.class_query.repeat(1, batch_size, 1)).transpose(0, 1)
        out = self.projection(transformer_out).squeeze()
        return out
    

# from net.utils.tgcn import ConvTemporalGraphical
class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
# from net.utils.graph import Graph
def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        layout = 'wlasl100'
        if layout == 'wlasl100':
            self.num_node = 54
            self_link0 = [(i, i) for i in range(self.num_node)]
            num_body_nodes = 12
            num_hand_nodes = 21
            # 定义连接关系
            body_edges = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
                (10, 11)
            ]
            
            # 左手的连接关系
            left_hand_edges = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            
            # 右手的连接关系
            right_hand_edges = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            
            # 将左手和右手的连接关系偏移
            left_hand_edges = [(i + num_body_nodes, j + num_body_nodes) for i, j in left_hand_edges]
            right_hand_edges = [(i + num_body_nodes + num_hand_nodes, j + num_body_nodes + num_hand_nodes) for i, j in right_hand_edges]
            
            # 合并所有连接关系
            # edges = body_edges + left_hand_edges + right_hand_edges
            self.edge = self_link0 + body_edges + left_hand_edges + right_hand_edges
            self.center = 20
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class,
                 edge_importance_weighting=True, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        # self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        # print(x.size())
        x = x.unsqueeze_(-1)
        # print(x.size())
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        # x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        # print(x.size())
        
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.1,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )


        self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

class SiFormer(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, num_hid=108, attn_type='prob', num_enc_layers=3, num_dec_layers=2, patience=1,
                 seq_len=204, device=None, IA_encoder = True, IA_decoder = False):
        super(SiFormer, self).__init__()
        print("Feature isolated transformer")
        self.gcn = Model(2,100)
        self.gcn.hidden = 128
        # self.feature_extractor = FeatureExtractor(num_hid=108, kernel_size=7)
        self.l_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.r_hand_embedding = nn.Parameter(self.get_encoding_table(d_model=42))
        self.body_embedding = nn.Parameter(self.get_encoding_table(d_model=24))

        self.class_query = nn.Parameter(torch.rand(1, 1, num_hid))
        self.transformer = FeatureIsolatedTransformer(
            [42, 42, 24], [3, 3, 2, 9], num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers,
            selected_attn=attn_type, IA_encoder=IA_encoder, IA_decoder=IA_decoder,
            inner_classifiers_config=[num_hid, num_classes], projections_config=[seq_len, 1],  device=device,
            patience=patience, use_pyramid_encoder=False, distil=False
        )
        print(f"num_enc_layers {num_enc_layers}, num_dec_layers {num_dec_layers}, patient {patience}")
        self.projection = nn.Linear(num_hid + self.gcn.hidden, num_classes)

    def forward(self, l_hand, r_hand, body, training):
        combined_data = torch.cat((body,l_hand, r_hand), dim=2)
        combined_data = combined_data.permute(0, 3, 1, 2).contiguous().float() 
        outgcn = self.gcn(combined_data)
        # print(outgcn.size())
        batch_size = l_hand.size(0)
        # (batch_size, seq_len, respected_feature_size, coordinates): (24, 204, 54, 2)
        # -> (batch_size, seq_len, feature_size):  (24, 204, 108)
        new_l_hand = l_hand.view(l_hand.size(0), l_hand.size(1), l_hand.size(2) * l_hand.size(3))
        new_r_hand = r_hand.view(r_hand.size(0), r_hand.size(1), r_hand.size(2) * r_hand.size(3))
        body = body.view(body.size(0), body.size(1), body.size(2) * body.size(3))

        # (batch_size, seq_len, feature_size) : (24, 204, 108)
        # -> (seq_len, batch_size, feature_size): (204, 24, 108)
        new_l_hand = new_l_hand.permute(1, 0, 2).type(dtype=torch.float32)
        new_r_hand = new_r_hand.permute(1, 0, 2).type(dtype=torch.float32)
        new_body = body.permute(1, 0, 2).type(dtype=torch.float32)

        # feature_map = self.feature_extractor(new_inputs)
        # transformer_in = feature_map + self.pos_embedding
        l_hand_in = new_l_hand + self.l_hand_embedding  # Shape remains the same
        r_hand_in = new_r_hand + self.r_hand_embedding  # Shape remains the same
        body_in = new_body + self.body_embedding  # Shape remains the same

        # (seq_len, batch_size, feature_size) -> (batch_size, 1, feature_size): (24, 1, 108)
        transformer_output = self.transformer(
            [l_hand_in, r_hand_in, body_in], self.class_query.repeat(1, batch_size, 1), training=training
        ).transpose(0, 1)
        # print("transformer_output.shape")
        # print(transformer_output.shape)

        outgcn = outgcn.unsqueeze_(1)
        combined_tensor = torch.cat((outgcn, transformer_output), dim=2)

        # (batch_size, 1, feature_size) -> (batch_size, num_class): (24, 100)
        out = self.projection(combined_tensor).squeeze()
        return out

    @staticmethod
    def get_encoding_table(d_model=108, seq_len=204):
        torch.manual_seed(42)
        tensor_shape = (seq_len, d_model)
        frame_pos = torch.rand(tensor_shape)
        for i in range(tensor_shape[0]):
            for j in range(1, tensor_shape[1]):
                frame_pos[i, j] = frame_pos[i, j - 1]
        frame_pos = frame_pos.unsqueeze(1)  # (seq_len, 1, feature_size): (204, 1, 108)
        return frame_pos
