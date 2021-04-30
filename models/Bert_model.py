import copy
import json
import math

import six
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from transformers.activations import gelu, gelu_new, swish
from models.layers import SharableLinear, SharableEmbedding


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}
BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = SharableEmbedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = SharableEmbedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = SharableEmbedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, multi_params=None, for_prfs=False):
        super().__init__()
        self.for_prfs = for_prfs
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        if multi_params is not None:
            self.num_attention_heads = multi_params
            self.attention_head_size = int(config.hidden_size_aug / self.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            hidden_size = config.hidden_size_aug
        else:
            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            hidden_size = config.hidden_size
        if not for_prfs:
            self.query = SharableLinear(hidden_size, self.all_head_size)
            self.key = SharableLinear(hidden_size, self.all_head_size)
            self.value = SharableLinear(hidden_size, self.all_head_size)
        else:
            self.query = nn.Linear(hidden_size, self.all_head_size)
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            sample=False
    ):
        if not self.for_prfs:
            mixed_query_layer = self.query(hidden_states, sample=sample)

            # If this is instantiated as a cross-attention module, the keys
            # and values come from an encoder; the attention mask needs to be
            # such that the encoder's padding tokens are not attended to.
            if encoder_hidden_states is not None:
                mixed_key_layer = self.key(encoder_hidden_states, sample=sample)
                mixed_value_layer = self.value(encoder_hidden_states, sample=sample)
                attention_mask = encoder_attention_mask
            else:
                mixed_key_layer = self.key(hidden_states, sample=sample)
                mixed_value_layer = self.value(hidden_states, sample=sample)
        else:
            mixed_query_layer = self.query(hidden_states)

            # If this is instantiated as a cross-attention module, the keys
            # and values come from an encoder; the attention mask needs to be
            # such that the encoder's padding tokens are not attended to.
            if encoder_hidden_states is not None:
                mixed_key_layer = self.key(encoder_hidden_states)
                mixed_value_layer = self.value(encoder_hidden_states)
                attention_mask = encoder_attention_mask
            else:
                mixed_key_layer = self.key(hidden_states)
                mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config, multi_params=None, houlsby=False):
        super().__init__()
        if houlsby:
            self.adapter = BertLowRank(config)
        if multi_params is not None:
            self.dense = nn.Linear(config.hidden_size_aug, config.hidden_size_aug)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = SharableLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.houlsby = houlsby

    def forward(self, hidden_states, input_tensor, attention_mask=None, sample=False):
        hidden_states = self.dense(hidden_states, sample=sample)
        hidden_states = self.dropout(hidden_states)
        if self.houlsby:
            hidden_states = hidden_states + self.adapter(hidden_states, attention_mask)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, multi_params=None, houlsby=False):
        super().__init__()
        self.self = BertSelfAttention(config, multi_params)
        self.output = BertSelfOutput(config, multi_params, houlsby)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            sample=False
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions, sample=sample
        )
        attention_output = self.output(self_outputs[0], hidden_states, attention_mask, sample=sample)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PRF(nn.Module):
    def __init__(self, config, extra_dim=None):
        super(PRF, self).__init__()
        # Encoder and decoder matrices project down to the smaller dimension
        self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
        self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
        # Attention without the final matrix multiply.
        self.attn = BertSelfAttention(config, config.prf_heads, for_prfs=True)
        self.config = config
        self.hidden_act_fn = gelu

    def forward(self, hidden_states, attention_mask=None):
        hidden_states_aug = self.aug_dense(hidden_states)
        hidden_states_aug = self.attn(hidden_states_aug, attention_mask)
        hidden_states = self.aug_dense2(hidden_states_aug[0])
        hidden_states = self.hidden_act_fn(hidden_states)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SharableLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, sample=False):
        hidden_states = self.dense(hidden_states, sample=sample)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertLhuc(nn.Module):
    def __init__(self, config):
        super(BertLhuc, self).__init__()
        self.lhuc = nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, hidden_states):
        hidden_states = hidden_states * 2. * nn.functional.sigmoid(self.lhuc)
        return hidden_states


class BertLowRank(nn.Module):
    def __init__(self, config, extra_dim=None):
        super(BertLowRank, self).__init__()
        # Encoder and decoder matrices project down to the smaller dimension
        if config.extra_dim:
            self.aug_dense = nn.Linear(config.hidden_size, config.extra_dim)
            self.aug_dense2 = nn.Linear(config.extra_dim, config.hidden_size)
        else:
            self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
            self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
        self.config = config
        self.hidden_act_fn = gelu

    def forward(self, hidden_states, attention_mask=None):
        hidden_states_aug = self.aug_dense(hidden_states)
        hidden_states_aug = self.hidden_act_fn(hidden_states_aug)
        hidden_states = self.aug_dense2(hidden_states_aug)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config, houlsby=False):
        super().__init__()
        self.dense = SharableLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if houlsby:
            if config.prfs:
                self.adapter = PRF(config)
            else:
                self.adapter = BertLowRank(config)
        self.houlsby = houlsby

    def forward(self, hidden_states, input_tensor, attention_mask=None, sample=False):
        hidden_states = self.dense(hidden_states, sample=sample)
        hidden_states = self.dropout(hidden_states)
        if self.houlsby:
            hidden_states = hidden_states + self.adapter(input_tensor, attention_mask)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, houlsby=False):
        super().__init__()
        self.attention = BertAttention(config, houlsby=houlsby)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config, houlsby=houlsby)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config, houlsby=houlsby)
        self.adapter = None
        if config.lhuc:
            self.adapter_lhuc = BertLhuc(config)
        if config.adapter:
            if config.prfs:
                self.adapter = PRF(config)
            else:
                self.adapter = BertLowRank(config)
        self.lhuc = config.lhuc
        self.houlsby = houlsby

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            sample=False
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions, sample=sample
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output, sample=sample)
        if self.lhuc and not self.adapter:
            layer_output = self.output(intermediate_output, attention_output, sample=sample)
            layer_output = self.adapter_lhuc(layer_output)
        elif self.adapter:
            extra = self.adapter(hidden_states, attention_mask)
            if self.lhuc:
                extra = self.adapter_lhuc(extra)
            layer_output = self.output(intermediate_output, attention_output + extra, sample=sample)
        elif self.houlsby:
            layer_output = self.output(intermediate_output, attention_output, attention_mask, sample=sample)
        else:
            layer_output = self.output(intermediate_output, attention_output, sample=sample)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.houlsby:
            self.layer = nn.ModuleList([BertLayer(config, houlsby=config.houlsby) for _ in range(config.num_hidden_layers)])
        else:
            self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        if config.adapter and config.prfs:
            self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
            # Shared encoder and decoder across layers
            self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
            for l, layer in enumerate(self.layer):
                layer.adapter.aug_dense = self.aug_dense
                layer.adapter.aug_dense2 = self.aug_dense2
        if config.houlsby and config.prfs:
            self.aug_dense = nn.Linear(config.hidden_size, config.hidden_size_aug)
            # Shared encoder and decoder across layers
            self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.hidden_size)
            for l, layer in enumerate(self.layer):
                layer.output.adapter.aug_dense = self.aug_dense
                layer.output.adapter.aug_dense2 = self.aug_dense2

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            sample=False
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    sample=sample
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = SharableLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.adapter = config.adapter

    def forward(self, hidden_states, sample=False):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor, sample)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            sample=False
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            sample=sample
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output, sample=sample)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:
                                                      ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 gradient_checkpointing=False,
                 adapter=True,
                 prfs=False,
                 prf_heads=6,
                 top=False,
                 lhuc=False,
                 houlsby=False,
                 bert_lay_top=False,
                 num_tasks=1,
                 extra_dim=None,
                 hidden_size_aug=204):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_size_aug = hidden_size_aug
        self.adapter = adapter
        self.prfs = prfs
        self.prf_heads = prf_heads
        self.extra_dim = extra_dim
        self.houlsby = houlsby
        self.top = top
        self.bert_lay_top = bert_lay_top
        self.lhuc = lhuc
        self.num_tasks = num_tasks

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
