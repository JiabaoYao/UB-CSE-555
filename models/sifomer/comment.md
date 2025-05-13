a.Siformer is mainly based on the FeatureIsolatedTransformer architecture, combining multiple feature extraction modules (such as left hand, right hand and body features) and the Transformer architecture for sequence modeling and classification.
Problem: Insufficient modeling of spatial structure information. Relying only on time series modeling may cause the network to fail to fully utilize the spatial information of the skeleton data, thus affecting the accuracy and robustness of classification.
b.Siformer can extract their respective time series features by inputting the left hand, right hand and body features into different encoders, but lacks fine modeling of local features (such as local movements of hand joints).

Improvement:
The convolutional neural network that introduces graph structure data can model spatial and temporal information at the same time. When the output of GCN is fused with the output of Transformer, the complementarity of global and local features can be more effectively utilized, thereby improving classification performance.
a. Enhanced spatial structure modeling capability

ST-GCN can fully utilize the spatial topological structure of skeleton data through graph convolution operations. It regards various parts of the body (such as joints) as nodes in the graph and the connection relationship between joints as edges, so that it can effectively capture the spatial relationship between various parts of the body.

b. Improve local feature capture capability
The graph convolution operation of ST-GCN can finely model local features. By defining a suitable adjacency matrix, ST-GCN can capture the motion information between local joints, thereby better processing complex movements in local areas such as hands.

c. More effective feature fusion
When fusing the output of ST-GCN with the output of Transformer, the model can simultaneously utilize the spatial information of graph structure data and the temporal information of time series. This combination of multimodal features can improve the robustness of the model to noise and interference.


To train the model:

```
python -m train
  --experiment_name [str; name of the experiment to name the output logs and plots]
  --epochs [int; number of epochs]
  --lr [float; learning rate]
  --num_classes [int; the number of classes to be recognised by the model]
  
  --attn_type [str; the attention mechanism used by the model]
  --num_enc_layers [int; determines the number of encoder layer]
  --num_dec_layers [int; determines the number of decoder layer]
  --FIM [boolean; determines whether feature-isolated mechanism will be applied]
  --PBEE_encoder [bool; determines whether patience-based encoder will be used for input-adaptive inference]
  --PBEE_decoder [bool; determines whether patience-based decoder will be used for input-adaptive inference]
  --patience [int; determines the patience for earlier exist]
  
  --training_set_path [str; the path to the CSV file with training set's skeletal data]
  --validation_set_path [str; the Path to the CSV file with validation set's skeletal data]
  --testing_set_path [str; the path to the CVS file with testing set's skeletal data]
```


Here are some examples of usage:
```
python -m train --experiment_name WLASL100 --training_set_path datasets/WLASL100_train_25fps.csv --validation_set_path datasets/WLASL100_val_25fps.csv --validation_set from-file --num_classes 100

```






@inproceedings{10.1145/3581783.3611724,
  author = {Muxin Pu, Mei Kuan Lim, and Chun Yong Chong},
  title = {Siformer: Feature-isolated Transformer for Efficient Skeleton-based Sign Language Recognition},
  year = {2024},
  isbn = {979-8-4007-0686-8},
  publisher = {Association for Computing Machinery},
  address = {Melbourne, VIC, Australia},
  url = {https://doi.org/10.1145/3664647.3681578},
  doi = {10.1145/3664647.3681578},
  booktitle = {Proceedings of the 32st ACM International Conference on Multimedia},
  series = {MM '24}
}
