import tensorflow as tf
import spektral
from spektral.layers import GCNConv,GatedGraphConv, GraphSageConv, GATConv, GlobalAvgPool, GlobalSumPool, ECCConv, SRCPool
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, GlobalAveragePooling1D,Dropout,GlobalMaxPooling1D, BatchNormalization, ReLU,  Concatenate, Masking
from .layer import EGT, MaskedGlobalAvgPooling2D, GetVirtualNodes, VirtualNodeEmbedding, VirtualEdgeEmbedding
from absl import flags
from tensorflow.keras import layers, models, regularizers

FLAGS = flags.FLAGS

class EGT_Model():
    def __init__(self, clip_logits_value=[-5, 5],scale_degree = False,gate_attention = 'True',edge_activation = None,add_n_norm = False,do_final_norm = True,edge_dropout = 0,edge_channel_type = 'residual',node_dropout = 0,
                ffn_multiplier = 1.0,activation = 'elu',node2edge_xtalk = 0.,edge2node_xtalk = 0., combine_layer_repr = False,return_all = False,readout_edges = True,mlp_layers = [1.0, 1.0],mask_value = -1,random_mask_prob = 0.1,
                attn_dropout = 0, scaler_type = 'log',use_adj = True,distance_loss = 0, distance_target = 8.,upto_hop = 16,clip_hops = True,include_xpose = False,max_degree_enc = 0,max_diffuse_t = 0):

        self.clip_logits_value = clip_logits_value
        self.scale_degree = scale_degree
        # self.max_length = max_length  # 360
        self.gate_attention = gate_attention
        self.edge_activation =edge_activation
        self.add_n_norm = add_n_norm
        self.do_final_norm = do_final_norm
        self.edge_dropout = edge_dropout
        self.edge_channel_type = edge_channel_type
        self.node_dropout = node_dropout
        self.ffn_multiplier =ffn_multiplier
        self.activation = activation
        self.node2edge_xtalk = node2edge_xtalk
        self.edge2node_xtalk = edge2node_xtalk
        self.combine_layer_repr = combine_layer_repr
        self.return_all = return_all
        self.readout_edges = readout_edges
        self.mlp_layers = mlp_layers
        self.mask_value = mask_value
        self.random_mask_prob = random_mask_prob
        self.attn_dropout = attn_dropout
        self.scaler_type = scaler_type
        self.use_adj = use_adj
        self.distance_loss = distance_loss
        self.distance_target = distance_target
        self.upto_hop = upto_hop
        self.clip_hops = clip_hops
        self.include_xpose = include_xpose
        self.max_degree_enc = max_degree_enc
        self.max_diffuse_t = max_diffuse_t


    def EGT_Backbone(self, node_dim, edge_dim , model_height, num_heads, num_virtual_nodes, max_length):

        l2reg = regularizers.l2(0)

        data_layer = tf.keras.layers.Input(shape=(max_length, node_dim))
        edge_layer = tf.keras.layers.Input(shape=(max_length, max_length, edge_dim))
        adj_layer = tf.keras.layers.Input(shape=(max_length, max_length))

        def create_embedding(name, x):

            if name == 'adj':
                if self.upto_hop == 1:
                    x = layers.Lambda(lambda v: v[..., None], name='adj_expand_dim')(x)
                elif self.upto_hop > 1:

                    def stack_hops(mat):
                        hops = [mat]
                        hop_mat = mat
                        for _ in range(self.upto_hop - 1):
                            hop_mat = tf.matmul(mat, hop_mat)
                            if self.clip_hops:
                                hop_mat = tf.clip_by_value(hop_mat, 0., 1.)
                            hops.append(hop_mat)
                        return tf.stack(hops, axis=-1)

                    x = Lambda(stack_hops, name='adj_stack_hops')(x)
                else:
                    raise ValueError

                if self.include_xpose:
                    x = Lambda(lambda v: tf.concat([v, tf.transpose(v, perm=[0, 2, 1, 3])], axis=-1),name='adj_include_transpose')(x)

                x = Dense(edge_dim, name='adj_emb',kernel_regularizer=l2reg)(x)

            return x

        def get_edge_mask(edge_inputs):

            if self.edge_channel_type == 'constrained':
                adj_mat = edge_inputs
                nh = num_heads
                edge_masks = layers.Lambda(lambda v: tf.tile(v[..., None], [1, 1, 1, nh]),name='adj_expand_mask')(adj_mat)
                return edge_masks
            else:
                edge_masks = None
                return edge_masks

        def get_edge_mask_v2(edge_inputs,edge_masks):

            if edge_masks is not None:
                num_nodes = num_virtual_nodes

                def expand_mask(e_mask):
                    bshape_d, eshape1_d, eshape2_d, nh_d = tf.unstack(tf.shape(e_mask))
                    row_true = tf.ones([bshape_d, num_nodes, eshape2_d, nh_d], dtype=e_mask.dtype)
                    col_true = tf.ones([bshape_d, eshape1_d + num_nodes, num_nodes, nh_d], dtype=e_mask.dtype)

                    e_mask = tf.concat([row_true, e_mask], axis=1)
                    e_mask = tf.concat([col_true, e_mask], axis=2)
                    return e_mask

                edge_masks = Lambda(expand_mask, name='virtual_node_expand_mask')(edge_masks)

            return edge_masks


        def node_stack(node_f):
            node_fs = tf.unstack(node_f, axis=-1)
            oh_vecs = []
            for feat, dim in zip(node_fs, node_ds):
                oh_vecs.append(tf.one_hot(feat, dim, dtype=tf.float32))
            node_oh = tf.concat(oh_vecs, axis=-1)
            return node_oh

        def edge_stack(edge_f):
            edge_fs = tf.unstack(edge_f, axis=-1)
            oh_vecs = []
            for feat, dim in zip(edge_fs, edge_ds):
                oh_vecs.append(tf.one_hot(feat, dim, dtype=tf.float32))
            edge_oh = tf.concat(oh_vecs, axis=-1)
            return edge_oh

        def compute_mask(inputs, mask):
            return mask

        ad = create_embedding('adj',adj_layer)
        edge_mask = get_edge_mask(adj_layer)
        edge_mask = get_edge_mask_v2(adj_layer,edge_mask)

        data_mask = Masking(mask_value=self.mask_value, name='node_mask')(data_layer)
        # data_layer = Lambda(node_stack, mask=compute_mask, name='node_feat_oh')(data_layer)
        data_mask = Dense(node_dim, name='node_emb',kernel_initializer='uniform',kernel_regularizer=l2reg)(data_mask)




        edge_feat = Masking(mask_value=self.mask_value, name='edge_feat_mask')(edge_layer)
        # edge_layer = layers.Lambda(edge_stack, mask=compute_mask, name='edge_feat_oh')(edge_layer)
        edge_feat = layers.Dense(edge_dim, name='edge_emb',kernel_initializer='uniform',kernel_regularizer=l2reg)(edge_feat)
        edge_feat = layers.Add(name='combine_edge')([edge_feat, ad])


        if num_virtual_nodes > 0:
            data_mask = VirtualNodeEmbedding(num_nodes=num_virtual_nodes, name='virtual_node_embedding')(data_mask)
            edge_feat = VirtualEdgeEmbedding(num_nodes=num_virtual_nodes,name='virtual_edge_embedding')(edge_feat)

        norm_dict = dict(
            layer=layers.LayerNormalization,
            batch=layers.BatchNormalization
        )
        normlr_node = norm_dict['layer']
        normlr_edge = norm_dict['layer']

        def edge_channel_contrib(tag, e):
            if self.edge_activation is not None and \
                    self.edge_activation.lower().startswith('lrelu'):
                alpha = float(self.edge_activation[-1]) / 10
                e = layers.Dense(num_heads, name=f'dense_edge_b_{tag}',
                                 activation=None,
                                 kernel_regularizer=l2reg)(e)
                e = layers.LeakyReLU(alpha=alpha, name=f'lrelu_edge_b_{tag}')(e)

            else:
                e = layers.Dense(num_heads, name=f'dense_edge_b_{tag}',
                                 activation=self.edge_activation,
                                 kernel_regularizer=l2reg)(e)
            return e

        def edge_update_none(tag, h, e):
            gates = None
            h, _ = mha_block(tag, h, e, gates)

            return h, e

        def edge_update_bias(tag, h, e):
            e0 = e
            gates = None
            if self.gate_attention:
                gates = layers.Dense(num_heads,
                                     activation=None,
                                     name=f'attention_gates_{tag}',
                                     kernel_regularizer=l2reg)(e)

            e = edge_channel_contrib(tag, e)

            h, e = mha_block(tag, h, e, gates)

            return h, e0

        def edge_update_residual(tag, h, e):
            y = e
            if not self.add_n_norm:
                e = normlr_edge(name=f'norm_edge_{tag}')(e)

            # all_edge_repr[tag] = e
            #
            gates = None
            if self.gate_attention:
                gates = layers.Dense(num_heads,
                                     activation=None,
                                     name=f'attention_gates_{tag}',
                                     kernel_regularizer=l2reg)(e)
                # Analysis

            e = edge_channel_contrib(tag, e)

            h, e = mha_block(tag, h, e, gates)

            e = layers.Dense(edge_dim, name=f'dense_edge_r_{tag}',
                             kernel_regularizer=l2reg)(e)
            if self.edge_dropout > 0:
                e = layers.Dropout(self.edge_dropout, name=f'drp_edge_{tag}')(e)
            e = layers.Add(name=f'res_edge_{tag}')([e, y])

            if self.add_n_norm:
                e = normlr_edge(name=f'norm_edge_{tag}')(e)

            return h, e

        def mha_block(tag, h, e, gates=None):
            y = h
            if not self.add_n_norm:
                h = normlr_node(name=f'norm_mha_{tag}')(h)


            qkv = layers.Dense(node_dim * 3, name=f'dense_qkv_{tag}',
                               kernel_regularizer=l2reg)(h)

            h, e, mat = EGT(num_heads=32,
                            clip_logits_value=self.clip_logits_value,
                            scale_degree=self.scale_degree,
                            edge_input=(self.edge_channel_type != 'none'),
                            gate_input=(gates is not None),
                            attn_mask=(edge_mask is not None),
                            name=f'mha_{tag}',
                            num_virtual_nodes=num_virtual_nodes,
                            random_mask_prob=self.random_mask_prob,
                            attn_dropout=self.attn_dropout,
                            scaler_type=self.scaler_type,
                            )([qkv] +
                              ([e]) +
                              ([gates] if gates is not None  else []) +
                              ([edge_mask] if edge_mask is not None    else []))
                # Analysis

            h = layers.Dense(node_dim, name=f'dense_mha_{tag}',
                             kernel_regularizer=l2reg)(h)
            if self.node_dropout > 0:
                h = layers.Dropout(0, name=f'drp_mha_{tag}')(h)
            h = layers.Add(name=f'res_mha_{tag}')([h, y])
            #
            if False:
                h = normlr_node(name=f'norm_mha_{tag}')(h)

            return h, e

        xtalk_flag = (self.node2edge_xtalk > 0. or
                      self.edge2node_xtalk > 0.)

        def ffnlr1(tag, x, width, normlr):
            y = x
            if not self.add_n_norm:
                x = normlr(name=f'norm_fnn_{tag}')(x)
            x = layers.Dense(round(width*self.ffn_multiplier),
                             activation=(self.activation
                                         if not xtalk_flag else None),
                             name=f'fnn_lr1_{tag}',
                             kernel_regularizer=l2reg)(x)
            return x, y

        def ffnact(tag, x):
            if xtalk_flag:
                return layers.Activation(self.activation,
                                        name=f'ffn_activ_{tag}')(x)
            else:
                return x

        def ffnlr2(tag, x, y, width, normlr, drpr):
            x = layers.Dense(width,
                             name=f'fnn_lr2_{tag}',
                             kernel_regularizer=l2reg)(x)
            if drpr>0:
                x=layers.Dropout(drpr, name=f'drp_fnn_{tag}')(x)
            x=layers.Add(name=f'res_fnn_{tag}')([x,y])

            if self.add_n_norm:
                x = normlr(name=f'norm_fnn_{tag}')(x)
            return x

        def channel_xtalk(tag, x_h, x_e):
            # node2edge_xtalk = node2edge_xtalk
            # edge2node_xtalk = edge2node_xtalk
            # ffn_multiplier = ffn_multiplier

            def xtalk_fn(inputs, mask):
                x_h, x_e = inputs
                m_h, _ = mask

                x_h_n = None
                if edge2node_xtalk > 0.:
                    nx_s = round(edge2node_xtalk * x_e.shape[-1] / self.ffn_multiplier)
                    nx_t = x_e.shape[-1] - nx_s * 2
                    x_er, x_ec, x_e = tf.split(x_e, [nx_s, nx_s, nx_t], axis=3)

                    m_h = tf.cast(m_h, x_h.dtype)
                    x_er = tf.reduce_sum(x_er * m_h[:, :, None, None], axis=1)
                    x_ec = tf.reduce_sum(x_ec * m_h[:, None, :, None], axis=2)

                    m_h_sum = tf.reduce_sum(m_h, axis=1)[:, None, None]
                    x_h_n = tf.math.divide_no_nan(x_er + x_ec, m_h_sum)

                    x_h_n.set_shape([None, None, nx_s])
                    x_e.set_shape([None, None, None, nx_t])

                x_e_n = None
                if self.node2edge_xtalk > 0.:
                    nx_s = round(self.node2edge_xtalk * x_h.shape[-1] / self.ffn_multiplier)
                    nx_t = x_h.shape[-1] - nx_s * 2
                    x_hr, x_hc, x_h = tf.split(x_h, [nx_s, nx_s, nx_t], axis=2)
                    x_e_n = x_hr[:, :, None, :] + x_hc[:, None, :, :]

                    x_e_n.set_shape([None, None, None, nx_s])
                    x_h.set_shape([None, None, nx_t])

                if x_h_n is not None:
                    x_h = tf.concat([x_h, x_h_n], axis=-1)
                if x_e_n is not None:
                    x_e = tf.concat([x_e, x_e_n], axis=-1)

                return x_h, x_e

            def compute_mask(inputs, mask):
                return mask

            if xtalk_flag:
                x_h, x_e = layers.Lambda(xtalk_fn, mask=compute_mask,
                                         name=f'xtalk_{tag}')([x_h, x_e])
            return x_h, x_e


        def ffn_block(tag, x_h, x_e):
            tag_h = 'node_' + tag
            x_h, y_h = ffnlr1(tag_h, x_h, node_dim, normlr_node)

            if self.edge_channel_type in ['residual', 'constrained']:
                tag_e = 'edge_' + tag
                x_e, y_e = ffnlr1(tag_e, x_e, edge_dim, normlr_edge)

                x_h, x_e = channel_xtalk(tag, x_h, x_e)

                x_e = ffnact(tag_e, x_e)
                x_e = ffnlr2(tag_e, x_e, y_e, edge_dim, normlr_edge, self.edge_dropout)

            x_h = ffnact(tag_h, x_h)
            x_h = ffnlr2(tag_h, x_h, y_h, node_dim, normlr_node, self.node_dropout)
            return x_h, x_e

        edge_update_fn_dict = dict(none = edge_update_none, constrained = edge_update_residual, bias = edge_update_bias, residual = edge_update_residual)
        edge_update = edge_update_fn_dict[self.edge_channel_type]

        for ii in range(model_height):
            ii_tag = f'{ii:0>2d}'
            if ii == 0:
                h, e = edge_update(ii_tag,data_mask,edge_feat)
                h, e = ffn_block(ii_tag, h, e)
            else:
                h,e = edge_update(ii_tag,h,e)
                h, e = ffn_block(ii_tag, h, e)

        if (not self.add_n_norm) and self.do_final_norm:
            h = normlr_node(name='node_norm_final')(h)
            if self.edge_channel_type in ['residual', 'constrained']:
                e = normlr_edge(name='edge_norm_final')(e)

        def add_additional_losses(additional_targets, h, e, h_all=None, e_all=None):
            num_nodes = num_virtual_nodes

            def crop_ec(inputs, mask=None):
                return inputs[:, num_nodes:, num_nodes:, :]

            def crop_mask(inputs, mask):
                return mask[:, num_nodes:, num_nodes:]

            e = Lambda(crop_ec, mask=crop_mask, name='crop_edge_channels')(e)
            return h, e

        h, e = add_additional_losses({}, h, e)

        if num_virtual_nodes > 0:
            h = GetVirtualNodes(num_nodes=num_virtual_nodes,
                                       name='get_virtual_nodes')(h)
            h = layers.Flatten(name='virtual_nodes_flatten')(h)
        else:
            h = layers.GlobalAveragePooling1D(name='node_glob_avg_pool')(h)

        x = h
        if self.readout_edges:
            e = MaskedGlobalAvgPooling2D(name='edge_glob_avg_pool')(e)
            x = Concatenate(name='cat_node_and_edge_out')([x,e])

        for ii,f in enumerate(self.mlp_layers):
            lr_name = f'mlp_out_{ii:0>1d}'
            x = Dense(round(f*node_dim),activation=self.activation,name=lr_name,kernel_regularizer=l2reg)(x)

        output = Dense(1, activation='sigmoid')(x)
        #
        model = Model(inputs=[data_layer, adj_layer, edge_layer], outputs=[output, x])


        return model
