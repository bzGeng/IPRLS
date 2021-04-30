import math
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from models.Bert_model import BertModel
import models.layers as nl


class BertForSequenceClassification(nn.Module):
    def __init__(self, dataset_history, dataset2num_classes, args):
        super(BertForSequenceClassification, self).__init__()
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.gama = args.gama
        self.config = args.configuration
        self.num_labels = self.config.num_labels
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.classifier = None
        self.bert = BertModel.from_pretrained(args.Bert_path, config=args.Bert_config_path)
        self.bert_saved = deepcopy(self.bert)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.dataset2num_classes = dataset2num_classes
        self.masks = None

        if self.datasets:
            self._reconstruct_classifiers()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            sample=False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            sample=sample
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, ) + outputs[2:]  # add hidden states and attention if they are here

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if self.args.mode == 'finetune' and len(self.datasets) > 1:
            prev_weight_strength = nn.Parameter(torch.Tensor(1, 1).uniform_(0, 0)).cuda()
            sigma_weight_reg_sum = 0
            mu_weight_reg_sum = 0
            L1_mu_weight_reg_sum = 0
            for (name, module), (name_saved, module_saved) in zip(self.bert.named_modules(), self.bert_saved.named_modules()):
                if isinstance(module, nl.SharableLinear) and (True in [ele in name for ele in self.args.shared_layers]):
                    assert name == name_saved
                    mask = self.masks[name]
                    trainer_weight = module.weight[mask.ne(self.current_dataset_idx)]
                    saver_weight = module_saved.weight[mask.ne(self.current_dataset_idx)]

                    out_features, in_features = module.weight.shape

                    trainer_weight_sigma = torch.log1p(torch.exp(module.weight_rho))
                    saver_weight_sigma = torch.log1p(torch.exp(module_saved.weight_rho))

                    std_init = math.sqrt((2 / out_features) * self.args.ratio)

                    saver_weight_strength = (std_init / saver_weight_sigma)

                    curr_strength = saver_weight_strength.expand(out_features, in_features)
                    prev_strength = prev_weight_strength.permute(1, 0).expand(out_features, in_features)

                    L2_strength = torch.max(curr_strength, prev_strength)[mask.ne(self.current_dataset_idx)]

                    trainer_weight_sigma = trainer_weight_sigma.expand(out_features, in_features)[mask.ne(self.current_dataset_idx)]
                    saver_weight_sigma = saver_weight_sigma.expand(out_features, in_features)[mask.ne(self.current_dataset_idx)]
                    L1_sigma = saver_weight_sigma

                    if 'key' in name:
                        prev_key_weight_strength = saver_weight_strength
                    elif 'query' in name:
                        prev_query_weight_strength = saver_weight_strength
                    elif 'value' in name:
                        prev_value_weight_strength = saver_weight_strength
                        prev_weight_strength = torch.max(torch.max(prev_key_weight_strength, prev_query_weight_strength), prev_value_weight_strength)
                    else:
                        prev_weight_strength = saver_weight_strength

                    mu_weight_reg = (L2_strength * (trainer_weight - saver_weight)).norm(2) ** 2

                    L1_mu_weight_reg = (torch.div(saver_weight ** 2, L1_sigma ** 2) * (
                                trainer_weight - saver_weight)).norm(1)
                    L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)

                    weight_sigma = (trainer_weight_sigma ** 2 / saver_weight_sigma ** 2)

                    sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
                    mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
                    L1_mu_weight_reg_sum = L1_mu_weight_reg_sum + L1_mu_weight_reg

            loss = loss / self.args.batch_size
            # L2 loss
            loss = loss + self.alpha * mu_weight_reg_sum / (2 * self.args.batch_size)
            # L1 loss
            loss = loss + self.beta * L1_mu_weight_reg_sum / self.args.batch_size
            # sigma regularization
            loss = loss + self.gama * sigma_weight_reg_sum / (2 * self.args.batch_size)
        outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(self.config.hidden_size, self.config.num_labels))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(self.config.hidden_size, self.config.num_labels))


def IPRLS(dataset_history, dataset2num_classes, args):
    return BertForSequenceClassification(dataset_history=dataset_history, dataset2num_classes=dataset2num_classes,
                                         args=args)
