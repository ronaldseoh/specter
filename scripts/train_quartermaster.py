import argparse
import os
import pathlib
import json

import torch
import pytorch_lightning as pl
import transformers

# data loader classes defined in the Lightning version of specter
from pytorch_lightning_training_script.train import (
    TripletLoss
)

import utils


ARG_TO_SCHEDULER = {
    "linear": transformers.optimization.get_linear_schedule_with_warmup,
    "cosine": transformers.optimization.get_cosine_schedule_with_warmup,
    "cosine_w_restarts": transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": transformers.optimization.get_polynomial_decay_schedule_with_warmup,
}

ARG_TO_SCHEDULER_CHOICES = sorted(ARG_TO_SCHEDULER.keys())
ARG_TO_SCHEDULER_METAVAR = "{" + ", ".join(ARG_TO_SCHEDULER_CHOICES) + "}"


class MultiFacetTripletLoss(torch.nn.Module):
    """
    Triplet loss function for multi-facet embeddings: Based on the TripletLoss function from  https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/model.py#L159
    """
    def __init__(self, loss_type="margin", margin=1.0, distance='l2-norm',
                 reduction='mean', reduction_multifacet='mean', reduction_multifacet_target=None, reduction_multifacet_query_first=False):
        """
        Args:
            margin: margin (float, optional): Default: `1`.
            distance: can be `l2-norm` or `cosine`, or `dot`
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        """
        super().__init__()

        self.loss_type = loss_type
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
        self.reduction_multifacet = reduction_multifacet
        self.reduction_multifacet_target = reduction_multifacet_target
        self.reduction_multifacet_query_first = reduction_multifacet_query_first

        if self.reduction_multifacet_query_first:
            self.reduction_multifacet_option_1 = self.reduction_multifacet
            self.reduction_multifacet_option_2 = self.reduction_multifacet_target
            self.reduction_multifacet_dimension_1 = 1
        else:
            self.reduction_multifacet_option_1 = self.reduction_multifacet_target
            self.reduction_multifacet_option_2 = self.reduction_multifacet
            self.reduction_multifacet_dimension_1 = 2

        if self.reduction_multifacet_option_2 is None:
            raise Exception("reduction_multifacet_option_2 should not be None.")

    def forward(self, query, positive, negative):
        # Are there any zero vectors in `query`, `positive`, and `negative`?
        # We need to introduce masks to filter them out.
        query_mask = torch.nan_to_num((query.sum(dim=2) == 0) * float('inf'), nan=1.0, posinf=float('inf'), neginf=float('-inf'))
        positive_mask = torch.nan_to_num((positive.sum(dim=2) == 0) * float('inf'), nan=1.0, posinf=float('inf'), neginf=float('-inf'))
        negative_mask = torch.nan_to_num((negative.sum(dim=2) == 0) * float('inf'), nan=1.0, posinf=float('inf'), neginf=float('-inf'))

        query_mask = query_mask.unsqueeze(-1).expand(query.size())
        positive_mask = positive_mask.unsqueeze(-1).expand(positive.size())
        negative_mask = negative_mask.unsqueeze(-1).expand(negative.size())

        # Since 0 * float('inf') == nan, zero vectors in _masked matrices would be now
        # the vectors of 'nan'
        query_masked = query * query_mask
        positive_masked = positive * positive_mask
        negative_masked = negative * negative_mask

        if self.distance == 'l2-norm':
            distance_positive_all_temp = torch.cdist(query_masked, positive_masked, p=2)
            distance_negative_all_temp = torch.cdist(query_masked, negative_masked, p=2)

            # Unlike bmm() below, cdist() outputs would have in-place operation autograd anomalies
            # without having the separate tensors created with clone()
            distance_positive_all = torch.clone(distance_positive_all_temp)
            distance_negative_all = torch.clone(distance_negative_all_temp)
        elif self.distance == 'dot':
            distance_positive_all = torch.bmm(query_masked, torch.transpose(positive_masked, 1, 2))
            distance_negative_all = torch.bmm(query_masked, torch.transpose(negative_masked, 1, 2))
        else:
            raise TypeError(f"Unrecognized option for `distance`:{self.distance}")

        # Calculating the distances with `_masked` matrices above would lead to `nan` distances,
        # which we need to filter for each 'reduction' operation below.
        distance_positive_all_isnan = torch.isnan(distance_positive_all)
        distance_negative_all_isnan = torch.isnan(distance_negative_all)

        if self.reduction_multifacet_option_1 is None:
            # We filter out nan values based on the choice of 'reduction_multifacet_option_2'
            if self.reduction_multifacet_option_2 == 'min':
                distance_positive_all[distance_positive_all_isnan] = float('inf')
                distance_negative_all[distance_negative_all_isnan] = float('inf')
            elif self.reduction_multifacet_option_2 in ('max', 'logsumexp'):
                distance_positive_all[distance_positive_all_isnan] = float('-inf')
                distance_negative_all[distance_negative_all_isnan] = float('-inf')
            elif self.reduction_multifacet_option_2 == 'mean':
                distance_positive_all[distance_positive_all_isnan] = 0.0
                distance_negative_all[distance_negative_all_isnan] = 0.0

            distance_positive_all = distance_positive_all.flatten(start_dim=1)
            distance_negative_all = distance_negative_all.flatten(start_dim=1)
        elif self.reduction_multifacet_option_1 == 'min':
            # Since we first look for the minimum distance in dim 2, we make invalid distances
            # to have positive infinity values.
            distance_positive_all[distance_positive_all_isnan] = float('inf')
            distance_negative_all[distance_negative_all_isnan] = float('inf')

            distance_positive_all = torch.min(distance_positive_all, dim=self.reduction_multifacet_dimension_1).values
            distance_negative_all = torch.min(distance_negative_all, dim=self.reduction_multifacet_dimension_1).values
        elif self.reduction_multifacet_option_1 in ('max', 'logsumexp'):
            # Since we first look for the maximum distance in dim 2, we make invalid distances
            # to have **negative** infinity values.
            distance_positive_all[distance_positive_all_isnan] = float('-inf')
            distance_negative_all[distance_negative_all_isnan] = float('-inf')

            if self.reduction_multifacet_option_1 == 'max':
                distance_positive_all = torch.max(distance_positive_all, dim=self.reduction_multifacet_dimension_1).values
                distance_negative_all = torch.max(distance_negative_all, dim=self.reduction_multifacet_dimension_1).values
            elif self.reduction_multifacet_option_1 == 'logsumexp':
                distance_positive_all = torch.logsumexp(distance_positive_all, dim=self.reduction_multifacet_dimension_1)
                distance_negative_all = torch.logsumexp(distance_negative_all, dim=self.reduction_multifacet_dimension_1)
        elif self.reduction_multifacet_option_1 == 'mean':
            # Since we calculate mean distance across dim 2, we make invalid distances
            # to be 0.
            distance_positive_all[distance_positive_all_isnan] = 0.0
            distance_negative_all[distance_negative_all_isnan] = 0.0

            # We need to calculate mean values with just the non-zero values.
            distance_positive_all = torch.sum(distance_positive_all, dim=self.reduction_multifacet_dimension_1) / torch.count_nonzero(distance_positive_all, dim=self.reduction_multifacet_dimension_1)
            distance_negative_all = torch.sum(distance_negative_all, dim=self.reduction_multifacet_dimension_1) / torch.count_nonzero(distance_negative_all, dim=self.reduction_multifacet_dimension_1)

        # At this point, all the invalid values are not completely eliminated at this point
        # as it could be the zero vectors in the dimension associated with `option_2`.
        # Since these invalid values would be 'inf', we filter them out using
        # torch.isinf()
        distance_positive_all_isinf = torch.isinf(distance_positive_all)
        distance_negative_all_isinf = torch.isinf(distance_negative_all)

        if self.reduction_multifacet_option_2 == 'min':
            distance_positive_all[distance_positive_all_isinf] = float('inf')
            distance_negative_all[distance_negative_all_isinf] = float('inf')

            distance_positive = torch.min(distance_positive_all, dim=1).values
            distance_negative = torch.min(distance_negative_all, dim=1).values
        elif self.reduction_multifacet_option_2 in ('max', 'logsumexp'):
            distance_positive_all[distance_positive_all_isinf] = float('-inf')
            distance_negative_all[distance_negative_all_isinf] = float('-inf')

            if self.reduction_multifacet_option_2 == 'max':
                distance_positive = torch.max(distance_positive_all, dim=1).values
                distance_negative = torch.max(distance_negative_all, dim=1).values
            elif self.reduction_multifacet_option_2 == 'logsumexp':
                distance_positive = torch.logsumexp(distance_positive_all, dim=1)
                distance_negative = torch.logsumexp(distance_negative_all, dim=1)
        elif self.reduction_multifacet_option_2 == 'mean':
            distance_positive_all[distance_positive_all_isinf] = 0.0
            distance_negative_all[distance_negative_all_isinf] = 0.0

            distance_positive = torch.sum(distance_positive_all, dim=1) / torch.count_nonzero(distance_positive_all, dim=1)
            distance_negative = torch.sum(distance_negative_all, dim=1) / torch.count_nonzero(distance_negative_all, dim=1)

        if self.loss_type == "bce":
            distances_as_logits = torch.stack([distance_negative, distance_positive], axis=1)
            labels = torch.tensor([1 for _ in range(len(distance_positive))], device=distance_positive.device)

            losses = torch.nn.functional.cross_entropy(distances_as_logits, labels)
        else:
            losses = torch.nn.functional.relu(distance_positive - distance_negative + self.margin)

        if self.reduction == 'mean':
            return losses.mean()

        if self.reduction == 'sum':
            return losses.sum()

        if self.reduction == 'none':
            return losses

        raise TypeError(f"Unrecognized option for `reduction`:{self.reduction}")


class QuarterMaster(pl.LightningModule):

    def __init__(self, **kwargs):

        super().__init__()

        self.save_hyperparameters()

        if self.hparams.model_behavior == "quartermaster":
            if "add_perturb_embeddings" in self.hparams:
                add_perturb = self.hparams.add_perturb_embeddings
            else:
                add_perturb = False

            if "init_bert_layer_facet_layers" in self.hparams:
                init_bert_layer_facet_layers = self.hparams.init_bert_layer_facet_layers
            else:
                init_bert_layer_facet_layers = False

            if "add_bert_layer_facet_layers_alternate" in self.hparams:
                add_bert_layer_facet_layers_alternate = self.hparams.add_bert_layer_facet_layers_alternate
            else:
                add_bert_layer_facet_layers_alternate = False

            if "add_extra_facet_layers_after" in self.hparams:
                # self.hparams.add_extra_facet_layers_after could be None too, so check that first
                if self.hparams.add_extra_facet_layers_after and len(self.hparams.add_extra_facet_layers_after) > 0:
                    self.model = utils.BertModelWithExtraLinearLayersForMultiFacets.from_pretrained(
                        self.hparams.pretrained_model_name,
                        add_extra_facet_layers_after=self.hparams.add_extra_facet_layers_after,
                        num_facets=self.hparams.num_facets,
                        add_perturb_embeddings=add_perturb,
                        init_bert_layer_facet_layers=init_bert_layer_facet_layers,
                        add_bert_layer_facet_layers_alternate=add_bert_layer_facet_layers_alternate)
                else:
                    self.model = utils.BertModelWithExtraLinearLayersForMultiFacets.from_pretrained(
                        self.hparams.pretrained_model_name,
                        add_perturb_embeddings=add_perturb)
            else:
                self.model = transformers.BertModel.from_pretrained(self.hparams.pretrained_model_name)
        else:
            # SPECTER
            self.model = transformers.BertModel.from_pretrained(self.hparams.pretrained_model_name)

        # Extra linear layers on top of each facet embeddings
        self.extra_facet_layers = torch.nn.ModuleList()

        if "add_extra_facet_layers" in self.hparams:
            if self.hparams.add_extra_facet_layers:
                if "add_extra_facet_layers_initialize_with_identical_random_weights" in self.hparams:
                    if self.hparams.add_extra_facet_layers_initialize_with_identical_random_weights:
                        # These weights will be applied to all extra facet layers
                        extra_linear_weight = torch.randn_like(self.model.pooler.dense.weight.data)
                        extra_linear_bias = torch.randn_like(self.model.pooler.dense.bias.data)

                for _ in range(self.hparams.num_facets):
                    extra_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

                    if "add_extra_facet_layers_initialize_with_nsp_weights" in self.hparams:
                        if self.hparams.add_extra_facet_layers_initialize_with_nsp_weights:
                            extra_linear.weight.data = self.model.pooler.dense.weight.data.clone()
                            extra_linear.bias.data = self.model.pooler.dense.bias.data.clone()
                    elif "add_extra_facet_layers_initialize_with_identical_random_weights" in self.hparams:
                        if self.hparams.add_extra_facet_layers_initialize_with_identical_random_weights:
                            extra_linear.weight.data = extra_linear_weight
                            extra_linear.bias.data = extra_linear_bias

                    self.extra_facet_layers.append(extra_linear)

                if "add_extra_facet_layers_initialize_0_with_nsp_weights" in self.hparams:
                    if self.hparams.add_extra_facet_layers_initialize_0_with_nsp_weights:
                        # These weights will be applied to all extra facet layers
                        self.extra_facet_layers[0].weight.data = self.model.pooler.dense.weight.data.clone()
                        self.extra_facet_layers[0].bias.data = self.model.pooler.dense.bias.data.clone()

        # Extra linear layers on top of each facet embeddings
        self.extra_facet_layers_for_target = torch.nn.ModuleList()

        if "add_extra_facet_layers_for_target" in self.hparams:
            if self.hparams.add_extra_facet_layers_for_target:
                if "add_extra_facet_layers_initialize_differently_for_target" in self.hparams:
                    if self.hparams.add_extra_facet_layers_initialize_differently_for_target or "extra_linear_weight" not in locals():
                        # These weights will be applied to all extra facet layers
                        extra_linear_weight = torch.randn_like(self.model.pooler.dense.weight.data)
                        extra_linear_bias = torch.randn_like(self.model.pooler.dense.bias.data)

                for _ in range(self.hparams.num_facets):
                    extra_linear = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

                    if "add_extra_facet_layers_initialize_with_nsp_weights" in self.hparams:
                        if self.hparams.add_extra_facet_layers_initialize_with_nsp_weights:
                            extra_linear.weight.data = self.model.pooler.dense.weight.data.clone()
                            extra_linear.bias.data = self.model.pooler.dense.bias.data.clone()
                    # If instead of elif to allow mixing NSP for source and random for target
                    if "add_extra_facet_layers_initialize_with_identical_random_weights" in self.hparams:
                        if "extra_linear_weight" in locals():
                            extra_linear.weight.data = extra_linear_weight
                            extra_linear.bias.data = extra_linear_bias

                    self.extra_facet_layers_for_target.append(extra_linear)

                if "add_extra_facet_layers_initialize_0_with_nsp_weights" in self.hparams:
                    if self.hparams.add_extra_facet_layers_initialize_0_with_nsp_weights:
                        # These weights will be applied to all extra facet layers
                        self.extra_facet_layers_for_target[0].weight.data = self.model.pooler.dense.weight.data.clone()
                        self.extra_facet_layers_for_target[0].bias.data = self.model.pooler.dense.bias.data.clone()

        if "add_extra_facet_nonlinearity" in self.hparams and self.hparams.add_extra_facet_nonlinearity:
            self.extra_facet_nonlinearity = torch.nn.Tanh()

        self.add_extra_facet_layers_alternate = False

        if "add_extra_facet_layers_alternate" in self.hparams and self.hparams.num_facets > 1 and self.hparams.add_extra_facet_layers_alternate:
            self.add_extra_facet_layers_alternate = True

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.hparams.pretrained_model_name)
        self.tokenizer.model_max_length = self.model.config.max_position_embeddings

        self.hparams.seqlen = self.model.config.max_position_embeddings

        self.use_multiple_losses = False
        self.use_target_token_embs = False
        self.do_not_use_target_token_embs_mean = False
        self.use_target_token_embs_kmeans = False
        self.use_target_token_embs_input = False
        self.use_target_token_embs_normalize = False
        self.sum_into_single_embeddings = None

        if self.hparams.model_behavior == 'specter':
            self.loss = TripletLoss(
                margin=self.hparams.loss_margin,
                distance=self.hparams.loss_distance,
                reduction=self.hparams.loss_reduction)
        else:
            if "loss_config" in self.hparams and self.hparams.loss_config is not None:
                self.use_multiple_losses = True

            if self.use_multiple_losses:
                self.loss_list = []

                for loss_config in self.hparams.loss_config:
                    if 'use_target_token_embs' in loss_config.keys() and loss_config['use_target_token_embs']:
                        self.use_target_token_embs = True

                        if 'do_not_use_target_token_embs_mean' in loss_config.keys() and loss_config['do_not_use_target_token_embs_mean']:
                            self.do_not_use_target_token_embs_mean = True

                        if "use_target_token_embs_kmeans" in loss_config.keys() and loss_config["use_target_token_embs_kmeans"]:
                            self.use_target_token_embs_kmeans = True

                        if "use_target_token_embs_input" in loss_config.keys() and loss_config["use_target_token_embs_input"]:
                            self.use_target_token_embs_input = True

                        if "use_target_token_embs_normalize" in loss_config.keys() and loss_config["use_target_token_embs_normalize"]:
                            self.use_target_token_embs_normalize = True

                    if "sum_into_single_embeddings" in loss_config.keys() and loss_config["sum_into_single_embeddings"]:
                        self.sum_into_single_embeddings = "training_only"

                    self.loss_list.append(
                        MultiFacetTripletLoss(
                            loss_type=loss_config["loss_type"],
                            margin=loss_config["margin"],
                            distance=loss_config["distance"],
                            reduction=loss_config["reduction"],
                            reduction_multifacet=loss_config["reduction_multifacet"],
                            reduction_multifacet_target=None if "reduction_multifacet_target" not in loss_config.keys() else loss_config["reduction_multifacet_target"],
                            reduction_multifacet_query_first=False if "reduction_multifacet_query_first" not in loss_config.keys() else loss_config["reduction_multifacet_query_first"],
                        )
                    )
            else:
                if "loss_type" in self.hparams:
                    loss_type = self.hparams.loss_type
                else:
                    loss_type = "margin"

                if "loss_use_target_token_embs" in self.hparams:
                    self.use_target_token_embs = self.hparams.loss_use_target_token_embs

                    if "loss_do_not_use_target_token_embs_mean" in self.hparams:
                        self.do_not_use_target_token_embs_mean = self.hparams.loss_do_not_use_target_token_embs_mean

                    if "loss_use_target_token_embs_kmeans" in self.hparams:
                        self.use_target_token_embs_kmeans = self.hparams.loss_use_target_token_embs_kmeans

                    if "loss_use_target_token_embs_input" in self.hparams:
                        self.use_target_token_embs_input = self.hparams.loss_use_target_token_embs_input

                    if "loss_use_target_token_embs_normalize" in self.hparams:
                        self.use_target_token_embs_normalize = self.hparams.loss_use_target_token_embs_normalize

                self.loss = MultiFacetTripletLoss(
                    loss_type=loss_type,
                    margin=self.hparams.loss_margin,
                    distance=self.hparams.loss_distance,
                    reduction=self.hparams.loss_reduction,
                    reduction_multifacet=self.hparams.loss_reduction_multifacet,
                    reduction_multifacet_target=self.hparams.loss_reduction_multifacet_target,
                    reduction_multifacet_query_first=self.hparams.loss_reduction_multifacet_query_first,
                )

            if self.use_target_token_embs_input:
                # Extra linear layers on top of each facet embeddings
                # to be used only with input_emb loss function calc
                self.scaling_layers_for_input_embs = torch.nn.ModuleList()

                for _ in range(self.hparams.num_facets):
                    scaler = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

                    self.scaling_layers_for_input_embs.append(scaler)

            if "sum_into_single_embeddings" in self.hparams:
                # What if self.sum_into_single_embeddings was set in the previous lines
                # and collide with the value given in self.hparams?
                if self.sum_into_single_embeddings is not None and self.sum_into_single_embeddings.startswith('training') and not (self.hparams.sum_into_single_embeddings is not None and self.hparams.sum_into_single_embeddings.startswith("training")):
                    self.sum_into_single_embeddings = "training_and_inference"
                else:
                    self.sum_into_single_embeddings = self.hparams.sum_into_single_embeddings

        self.opt = None

    def forward(self, input_ids, token_type_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        source_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.hparams.model_behavior == 'specter':
            return source_output[1]
        else:
            source_embedding = source_output.last_hidden_state[:, 0:self.hparams.num_facets, :]

            # Extra facet layer
            # pass through the extra linear layers for each facets if enabled
            if len(self.extra_facet_layers) > 0:
                if self.add_extra_facet_layers_alternate:
                    extra_facet_layers_mean_weight = torch.stack([f.weight for f in self.extra_facet_layers], dim=0).mean(dim=0)
                    extra_facet_layers_alternate_weights = [f.weight - extra_facet_layers_mean_weight for f in self.extra_facet_layers]

                # Pass the source embeddings through extra linear layers
                for n in range(self.hparams.num_facets):
                    # Instead of passing embeddings through output linear layers independently,
                    # Make the actual linear weights to be (linear weight of this facet - avg of all output weights)
                    if self.add_extra_facet_layers_alternate:
                        source_embedding[:, n, :] = torch.nn.functional.linear(source_embedding[:, n, :], extra_facet_layers_alternate_weights[n], self.extra_facet_layers[n].bias)
                    else:
                        source_embedding[:, n, :] = self.extra_facet_layers[n](source_embedding[:, n, :])

            if self.hparams.add_extra_facet_nonlinearity:
                source_embedding = self.extra_facet_nonlinearity(source_embedding)

            if self.sum_into_single_embeddings is not None \
            and self.sum_into_single_embeddings in ("training_and_inference", "inference_only"):
                return torch.sum(source_embedding, dim=1).unsqueeze(1)
            else:
                return source_embedding

    def train_dataloader(self):
        dataset = torch.utils.data.BufferedShuffleDataset(
            utils.IterableDataSetMultiWorker(file_path=self.hparams.train_file, tokenizer=self.tokenizer, size=self.hparams.train_size, block_size=100, num_facets=self.hparams.num_facets),
            buffer_size=100)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        # Don't use BufferedShuffleDataset here.
        dataset = utils.IterableDataSetMultiWorker(file_path=self.hparams.val_file, tokenizer=self.tokenizer, size=self.hparams.val_size, block_size=100, num_facets=self.hparams.num_facets)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.total_gpus)  # TODO: consider num_tpu_cores

        effective_batch_size = self.hparams.batch_size * self.hparams.grad_accum * num_devices

        return (self.hparams.train_size / effective_batch_size) * self.hparams.num_epochs

    def get_lr_scheduler(self):
        get_schedule_func = ARG_TO_SCHEDULER[self.hparams.lr_scheduler]

        if self.opt is None:
            return Exception("get_lr_scheduler() should not be called before the optimizer is configured.")

        if self.hparams.lr_scheduler == "cosine_w_restarts":
            scheduler = get_schedule_func(
                self.opt,
                num_warmup_steps=int(self.hparams.warmup_frac * self.total_steps),
                num_training_steps=self.total_steps,
                num_cycles=3)
        else:
            scheduler = get_schedule_func(
                self.opt,
                num_warmup_steps=int(self.hparams.warmup_frac * self.total_steps),
                num_training_steps=self.total_steps)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.adafactor:
            optimizer = transformers.optimization.Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False)
        else:
            optimizer = transformers.AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)

        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def _get_normalized_embeddings(self, source_embedding, pos_embedding, neg_embedding):
        # Normalize each facet embeddings for visualization purposes
        source_embedding_normalized = torch.nn.functional.normalize(source_embedding, p=2, dim=-1)
        pos_embedding_normalized = torch.nn.functional.normalize(pos_embedding, p=2, dim=-1)
        neg_embedding_normalized = torch.nn.functional.normalize(neg_embedding, p=2, dim=-1)

        return source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized

    def _calculate_loss_set_reg(self, source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val, is_before_extra):
        source_batch_mean = torch.mean(source_embedding_normalized, dim=0, keepdims=True)
        source_loss_set_reg = torch.mean(torch.linalg.norm(source_embedding_normalized - source_batch_mean, ord=2, dim=-1))

        pos_batch_mean = torch.mean(pos_embedding_normalized, dim=0, keepdims=True)
        pos_loss_set_reg = torch.mean(torch.linalg.norm(pos_embedding_normalized - pos_batch_mean, ord=2, dim=-1))

        neg_batch_mean = torch.mean(neg_embedding_normalized, dim=0, keepdims=True)
        neg_loss_set_reg = torch.mean(torch.linalg.norm(neg_embedding_normalized - neg_batch_mean, ord=2, dim=-1))

        val_string = ""

        if is_val:
            val_string = "val_"

        before_extra_string = ""

        if is_before_extra:
            before_extra_string = "_before_extra"

        self.log(
            val_string + "source_loss_set_reg" + before_extra_string, source_loss_set_reg,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.log(
            val_string + "pos_loss_set_reg" + before_extra_string, pos_loss_set_reg,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.log(
            val_string + "neg_loss_set_reg" + before_extra_string, neg_loss_set_reg,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def _calculate_facet_distances_mean(self, source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val, is_before_extra):
        source_facets_center_point = torch.mean(source_embedding_normalized, dim=1, keepdims=True)
        source_facets_distances_mean = torch.mean(torch.linalg.norm(source_embedding_normalized - source_facets_center_point, ord=2, dim=-1))

        pos_facets_center_point = torch.mean(pos_embedding_normalized, dim=1, keepdims=True)
        pos_facets_distances_mean = torch.mean(torch.linalg.norm(pos_embedding_normalized - pos_facets_center_point, ord=2, dim=-1))

        neg_facets_center_point = torch.mean(neg_embedding_normalized, dim=1, keepdims=True)
        neg_facets_distances_mean = torch.mean(torch.linalg.norm(neg_embedding_normalized - neg_facets_center_point, ord=2, dim=-1))

        val_string = ""

        if is_val:
            val_string = "val_"

        before_extra_string = ""

        if is_before_extra:
            before_extra_string = "_before_extra"

        self.log(
            val_string + "source_facets_distances_mean" + before_extra_string, source_facets_distances_mean,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.log(
            val_string + "pos_facets_distances_mean" + before_extra_string, pos_facets_distances_mean,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

        self.log(
            val_string + "neg_facets_distances_mean" + before_extra_string, neg_facets_distances_mean,
            on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        if self.hparams.model_behavior == 'specter':
            # [1] actually contains what's referred to as "pooled output" in the Huggingface docs,
            # which is the [CLS] last hidden state followed by the BERT NSP linear layer
            source_embedding = self.model(
                input_ids=batch[0]['input_ids'], token_type_ids=batch[0]['token_type_ids'], attention_mask=batch[0]['attention_mask'])[1]
            pos_embedding = self.model(
                input_ids=batch[1]['input_ids'], token_type_ids=batch[1]['token_type_ids'], attention_mask=batch[1]['attention_mask'])[1]
            neg_embedding = self.model(
                input_ids=batch[2]['input_ids'], token_type_ids=batch[2]['token_type_ids'], attention_mask=batch[2]['attention_mask'])[1]

            with torch.no_grad():
                # Normalize each facet embeddings for visualization purposes
                source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized = self._get_normalized_embeddings(source_embedding, pos_embedding, neg_embedding)

                self._calculate_loss_set_reg(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=False, is_before_extra=False)
        else:
            source_output = self.model(
                input_ids=batch[0]['input_ids'], token_type_ids=batch[0]['token_type_ids'], attention_mask=batch[0]['attention_mask'])
            pos_output = self.model(
                input_ids=batch[1]['input_ids'], token_type_ids=batch[1]['token_type_ids'], attention_mask=batch[1]['attention_mask'])
            neg_output = self.model(
                input_ids=batch[2]['input_ids'], token_type_ids=batch[2]['token_type_ids'], attention_mask=batch[2]['attention_mask'])

            source_embedding = source_output.last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()
            pos_embedding = pos_output.last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()
            neg_embedding = neg_output.last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()

            if self.use_target_token_embs:
                # Use 'special_tokens_mask' to sum up just the sequence tokens and exclude all special tokens
                # [CLS], [unused0~k] (for extra facets), [SEP], [PAD]
                # Note for `special_tokens_mask', 0 for sequence tokens and 1 for special tokens.
                # Since we want special tokens to be filtered out, we first need to invert this mask tensor.
                pos_special_tokens_mask_inverted = ~(batch[1]['special_tokens_mask'].type(torch.bool))
                neg_special_tokens_mask_inverted = ~(batch[2]['special_tokens_mask'].type(torch.bool))

                # Since the masks above would have the dimensions of (batch size, 512), we use unsqueeze() and expand()
                # to match the shape of last_hidden_state, which would have the shape of (batch size, 512, 768).
                pos_special_tokens_mask_inverted_expanded = pos_special_tokens_mask_inverted.unsqueeze(-1).expand(pos_output.last_hidden_state.size())
                neg_special_tokens_mask_inverted_expanded = neg_special_tokens_mask_inverted.unsqueeze(-1).expand(neg_output.last_hidden_state.size())

                # Multiply the last hidden states and corresponding masks elementwise,
                if self.use_target_token_embs_input:
                    pos_embedding_tokens = self.model.get_input_embeddings()(batch[1]['input_ids'])
                    neg_embedding_tokens = self.model.get_input_embeddings()(batch[2]['input_ids'])
                else:
                    pos_embedding_tokens = pos_output.last_hidden_state
                    neg_embedding_tokens = neg_output.last_hidden_state

                pos_embedding_tokens = pos_embedding_tokens * pos_special_tokens_mask_inverted_expanded
                neg_embedding_tokens = neg_embedding_tokens * neg_special_tokens_mask_inverted_expanded

                if self.use_target_token_embs_normalize:
                    pos_embedding_tokens_normalized = torch.nn.functional.normalize(pos_embedding_tokens, p=2, dim=-1)
                    neg_embedding_tokens_normalized = torch.nn.functional.normalize(neg_embedding_tokens, p=2, dim=-1)

                # sum across dimension 1, then divide them by the number of non-zero elements
                if not self.do_not_use_target_token_embs_mean:
                    pos_embedding_tokens_mean = pos_embedding_tokens.sum(dim=1) / torch.count_nonzero(pos_embedding_tokens, dim=1)
                    pos_embedding_tokens_mean = pos_embedding_tokens_mean.unsqueeze(1)

                    neg_embedding_tokens_mean = neg_embedding_tokens.sum(dim=1) / torch.count_nonzero(neg_embedding_tokens, dim=1)
                    neg_embedding_tokens_mean = neg_embedding_tokens_mean.unsqueeze(1)

                if self.use_target_token_embs_kmeans:
                    pos_embedding_tokens_kmeans = utils.batch_k_means_cosine(pos_output.last_hidden_state, self.hparams.num_facets, 50, pos_special_tokens_mask_inverted)
                    neg_embedding_tokens_kmeans = utils.batch_k_means_cosine(neg_output.last_hidden_state, self.hparams.num_facets, 50, neg_special_tokens_mask_inverted)

            # pass through the extra linear layers for each facets if enabled
            if len(self.extra_facet_layers) > 0:
                # Before passing embeddings through extra facet layers,
                # measure loss_set_reg and facet_distances
                with torch.no_grad():
                    # Normalize each facet embeddings for visualization purposes
                    source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized = self._get_normalized_embeddings(source_embedding, pos_embedding, neg_embedding)

                    self._calculate_loss_set_reg(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=False, is_before_extra=True)

                    self._calculate_facet_distances_mean(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=False, is_before_extra=True)

                if self.add_extra_facet_layers_alternate:
                    extra_facet_layers_mean_weight = torch.stack([f.weight for f in self.extra_facet_layers], dim=0).mean(dim=0)
                    extra_facet_layers_alternate_weights = [f.weight - extra_facet_layers_mean_weight for f in self.extra_facet_layers]

                # Pass the source embeddings through extra linear layers
                for n in range(self.hparams.num_facets):
                    # Instead of passing embeddings through output linear layers independently,
                    # Make the actual linear weights to be (linear weight of this facet - avg of all output weights)
                    if self.add_extra_facet_layers_alternate:
                        source_embedding[:, n, :] = torch.nn.functional.linear(source_embedding[:, n, :], extra_facet_layers_alternate_weights[n], self.extra_facet_layers[n].bias)
                    else:
                        source_embedding[:, n, :] = self.extra_facet_layers[n](source_embedding[:, n, :])

            # For positive and negative papers, we could choose to train separate linear layers from source/pos
            if len(self.extra_facet_layers_for_target) > 0:
                # Instead of passing embeddings through output linear layers independently,
                # Make the actual linear weights to be (linear weight of this facet - avg of all output weights)
                if self.add_extra_facet_layers_alternate:
                    extra_facet_layers_for_target_mean_weight = torch.stack([f.weight for f in self.extra_facet_layers_for_target], dim=0).mean(dim=0)
                    extra_facet_layers_for_target_alternate_weights = [f.weight - extra_facet_layers_mean_weight for f in self.extra_facet_layers_for_target]

                for n in range(self.hparams.num_facets):
                    if self.add_extra_facet_layers_alternate:
                        pos_embedding[:, n, :] = torch.nn.functional.linear(pos_embedding[:, n, :], extra_facet_layers_for_target_alternate_weights[n], self.extra_facet_layers_for_target[n].bias)
                        neg_embedding[:, n, :] = torch.nn.functional.linear(neg_embedding[:, n, :], extra_facet_layers_for_target_alternate_weights[n], self.extra_facet_layers_for_target[n].bias)
                    else:
                        pos_embedding[:, n, :] = self.extra_facet_layers_for_target[n](pos_embedding[:, n, :])
                        neg_embedding[:, n, :] = self.extra_facet_layers_for_target[n](neg_embedding[:, n, :])
            elif len(self.extra_facet_layers) > 0:
                for n in range(self.hparams.num_facets):
                    if self.add_extra_facet_layers_alternate:
                        pos_embedding[:, n, :] = torch.nn.functional.linear(pos_embedding[:, n, :], extra_facet_layers_alternate_weights[n], self.extra_facet_layers[n].bias)
                        neg_embedding[:, n, :] = torch.nn.functional.linear(neg_embedding[:, n, :], extra_facet_layers_alternate_weights[n], self.extra_facet_layers[n].bias)
                    else:
                        pos_embedding[:, n, :] = self.extra_facet_layers[n](pos_embedding[:, n, :])
                        neg_embedding[:, n, :] = self.extra_facet_layers[n](neg_embedding[:, n, :])

            if self.hparams.add_extra_facet_nonlinearity:
                source_embedding = self.extra_facet_nonlinearity(source_embedding)
                pos_embedding = self.extra_facet_nonlinearity(pos_embedding)
                neg_embedding = self.extra_facet_nonlinearity(neg_embedding)

            # Normalize each facet embeddings
            source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized = self._get_normalized_embeddings(source_embedding, pos_embedding, neg_embedding)

            with torch.no_grad():
                self._calculate_loss_set_reg(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=False, is_before_extra=False)

                if self.hparams.num_facets > 1:
                    self._calculate_facet_distances_mean(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=False, is_before_extra=False)

            if self.sum_into_single_embeddings is not None \
            and self.sum_into_single_embeddings in ("training_and_inference", "training_only"):
                source_embedding_summed = torch.sum(source_embedding, dim=1).unsqueeze(1)
                pos_embedding_summed = torch.sum(pos_embedding, dim=1).unsqueeze(1)
                neg_embedding_summed = torch.sum(neg_embedding, dim=1).unsqueeze(1)

            if self.use_target_token_embs_input:
                # Pass the source embeddings through extra scaling layers
                source_embedding_scaled = torch.clone(source_embedding)

                for n in range(self.hparams.num_facets):
                    source_embedding_scaled[:, n, :] = self.scaling_layers_for_input_embs[n](source_embedding_scaled[:, n, :])

                if self.use_target_token_embs_normalize:
                    source_embedding_scaled_normalized = torch.nn.functional.normalize(source_embedding_scaled, p=2, dim=-1)

        if self.use_multiple_losses:
            loss = 0

            for i, l in enumerate(self.loss_list):
                if 'use_target_token_embs' in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]['use_target_token_embs']:
                    if "use_target_token_embs_kmeans" in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]["use_target_token_embs_kmeans"]:
                        this_loss = l(source_embedding, pos_embedding_tokens_kmeans, neg_embedding_tokens_kmeans)
                    elif 'do_not_use_target_token_embs_mean' in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]['do_not_use_target_token_embs_mean']:
                        if self.use_target_token_embs_normalize:
                            if self.use_target_token_embs_input:
                                this_loss = l(source_embedding_scaled_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                            else:
                                this_loss = l(source_embedding_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                        else:
                            if self.use_target_token_embs_input:
                                this_loss = l(source_embedding_scaled, pos_embedding_tokens, neg_embedding_tokens)
                            else:
                                this_loss = l(source_embedding, pos_embedding_tokens, neg_embedding_tokens)
                    else:
                        this_loss = l(source_embedding, pos_embedding_tokens_mean, neg_embedding_tokens_mean)
                elif "sum_into_single_embeddings" in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]['sum_into_single_embeddings']:
                    this_loss = l(source_embedding_summed, pos_embedding_summed, neg_embedding_summed)
                else:
                    this_loss = l(source_embedding, pos_embedding, neg_embedding)

                self.log('train_loss_' + self.hparams.loss_config[i]["name"], this_loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)

                loss = loss + self.hparams.loss_config[i]["weight"] * this_loss
        else:
            if "loss_use_target_token_embs" in self.hparams and self.hparams.loss_use_target_token_embs:
                if "loss_use_target_token_embs_kmeans" in self.hparams and self.hparams.loss_use_target_token_embs_kmeans:
                    loss = self.loss(source_embedding, pos_embedding_tokens_kmeans, neg_embedding_tokens_kmeans)
                elif "loss_do_not_use_target_token_embs_mean" in self.hparams and self.hparams.loss_do_not_use_target_token_embs_mean:
                    if self.use_target_token_embs_normalize:
                        if self.use_target_token_embs_input:
                            loss = self.loss(source_embedding_scaled_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                        else:
                            loss = self.loss(source_embedding_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                    else:
                        if self.use_target_token_embs_input:
                            loss = self.loss(source_embedding_scaled, pos_embedding_tokens, neg_embedding_tokens)
                        else:
                            loss = self.loss(source_embedding, pos_embedding_tokens, neg_embedding_tokens)
                else:
                    loss = self.loss(source_embedding, pos_embedding_tokens_mean, neg_embedding_tokens_mean)
            elif self.sum_into_single_embeddings is not None and self.sum_into_single_embeddings in ("training_and_inference", "training_only"):
                loss = self.loss(source_embedding_summed, pos_embedding_summed, neg_embedding_summed)
            else:
                loss = self.loss(source_embedding, pos_embedding, neg_embedding)

            self.log('train_loss_original', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)

        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.model_behavior == 'specter':
            # [1] actually contains what's referred to as "pooled output" in the Huggingface docs,
            # which is the [CLS] last hidden state followed by the BERT NSP linear layer
            source_embedding = self.model(
                input_ids=batch[0]['input_ids'], token_type_ids=batch[0]['token_type_ids'], attention_mask=batch[0]['attention_mask'])[1]
            pos_embedding = self.model(
                input_ids=batch[1]['input_ids'], token_type_ids=batch[1]['token_type_ids'], attention_mask=batch[1]['attention_mask'])[1]
            neg_embedding = self.model(
                input_ids=batch[2]['input_ids'], token_type_ids=batch[2]['token_type_ids'], attention_mask=batch[2]['attention_mask'])[1]

            # Normalize each facet embeddings for visualization purposes
            source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized = self._get_normalized_embeddings(source_embedding, pos_embedding, neg_embedding)

            self._calculate_loss_set_reg(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=True, is_before_extra=False)
        else:
            source_output = self.model(
                input_ids=batch[0]['input_ids'], token_type_ids=batch[0]['token_type_ids'], attention_mask=batch[0]['attention_mask'])
            pos_output = self.model(
                input_ids=batch[1]['input_ids'], token_type_ids=batch[1]['token_type_ids'], attention_mask=batch[1]['attention_mask'])
            neg_output = self.model(
                input_ids=batch[2]['input_ids'], token_type_ids=batch[2]['token_type_ids'], attention_mask=batch[2]['attention_mask'])

            source_embedding = source_output.last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()
            pos_embedding = pos_output.last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()
            neg_embedding = neg_output.last_hidden_state[:, 0:self.hparams.num_facets, :].contiguous()

            if self.use_target_token_embs:
                # Use 'special_tokens_mask' to sum up just the sequence tokens and exclude all special tokens
                # [CLS], [unused0~k] (for extra facets), [SEP], [PAD]
                # Note for `special_tokens_mask', 0 for sequence tokens and 1 for special tokens.
                # Since we want special tokens to be filtered out, we first need to invert this mask tensor.
                pos_special_tokens_mask_inverted = ~(batch[1]['special_tokens_mask'].type(torch.bool))
                neg_special_tokens_mask_inverted = ~(batch[2]['special_tokens_mask'].type(torch.bool))

                # Since the masks above would have the dimensions of (batch size, 512), we use unsqueeze() and expand()
                # to match the shape of last_hidden_state, which would have the shape of (batch size, 512, 768).
                pos_special_tokens_mask_inverted_expanded = pos_special_tokens_mask_inverted.unsqueeze(-1).expand(pos_output.last_hidden_state.size())
                neg_special_tokens_mask_inverted_expanded = neg_special_tokens_mask_inverted.unsqueeze(-1).expand(neg_output.last_hidden_state.size())

                # Multiply the last hidden states and corresponding masks elementwise,
                if self.use_target_token_embs_input:
                    pos_embedding_tokens = self.model.get_input_embeddings()(batch[1]['input_ids'])
                    neg_embedding_tokens = self.model.get_input_embeddings()(batch[2]['input_ids'])
                else:
                    pos_embedding_tokens = pos_output.last_hidden_state
                    neg_embedding_tokens = neg_output.last_hidden_state

                pos_embedding_tokens = pos_embedding_tokens * pos_special_tokens_mask_inverted_expanded
                neg_embedding_tokens = neg_embedding_tokens * neg_special_tokens_mask_inverted_expanded

                if self.use_target_token_embs_normalize:
                    pos_embedding_tokens_normalized = torch.nn.functional.normalize(pos_embedding_tokens, p=2, dim=-1)
                    neg_embedding_tokens_normalized = torch.nn.functional.normalize(neg_embedding_tokens, p=2, dim=-1)

                # sum across dimension 1, then divide them by the number of non-zero elements
                if not self.do_not_use_target_token_embs_mean:
                    pos_embedding_tokens_mean = pos_embedding_tokens.sum(dim=1) / torch.count_nonzero(pos_embedding_tokens, dim=1)
                    pos_embedding_tokens_mean = pos_embedding_tokens_mean.unsqueeze(1)

                    neg_embedding_tokens_mean = neg_embedding_tokens.sum(dim=1) / torch.count_nonzero(neg_embedding_tokens, dim=1)
                    neg_embedding_tokens_mean = neg_embedding_tokens_mean.unsqueeze(1)

                if self.use_target_token_embs_kmeans:
                    pos_embedding_tokens_kmeans = utils.batch_k_means_cosine(pos_output.last_hidden_state, self.hparams.num_facets, 50, pos_special_tokens_mask_inverted)
                    neg_embedding_tokens_kmeans = utils.batch_k_means_cosine(neg_output.last_hidden_state, self.hparams.num_facets, 50, neg_special_tokens_mask_inverted)

            # pass through the extra linear layers for each facets if enabled
            if len(self.extra_facet_layers) > 0:
                # Before passing embeddings through extra facet layers,
                # measure loss_set_reg and facet_distances
                # Normalize each facet embeddings for visualization purposes
                source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized = self._get_normalized_embeddings(source_embedding, pos_embedding, neg_embedding)

                self._calculate_loss_set_reg(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=True, is_before_extra=True)

                self._calculate_facet_distances_mean(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=True, is_before_extra=True)

                if self.add_extra_facet_layers_alternate:
                    extra_facet_layers_mean_weight = torch.stack([f.weight for f in self.extra_facet_layers], dim=0).mean(dim=0)
                    extra_facet_layers_alternate_weights = [f.weight - extra_facet_layers_mean_weight for f in self.extra_facet_layers]

                # Pass the source embeddings through extra linear layers
                for n in range(self.hparams.num_facets):
                    # Instead of passing embeddings through output linear layers independently,
                    # Make the actual linear weights to be (linear weight of this facet - avg of all output weights)
                    if self.add_extra_facet_layers_alternate:
                        source_embedding[:, n, :] = torch.nn.functional.linear(source_embedding[:, n, :], extra_facet_layers_alternate_weights[n], self.extra_facet_layers[n].bias)
                    else:
                        source_embedding[:, n, :] = self.extra_facet_layers[n](source_embedding[:, n, :])

            # For positive and negative papers, we could choose to train separate linear layers from source/pos
            if len(self.extra_facet_layers_for_target) > 0:
                # Instead of passing embeddings through output linear layers independently,
                # Make the actual linear weights to be (linear weight of this facet - avg of all output weights)
                if self.add_extra_facet_layers_alternate:
                    extra_facet_layers_for_target_mean_weight = torch.stack([f.weight for f in self.extra_facet_layers_for_target], dim=0).mean(dim=0)
                    extra_facet_layers_for_target_alternate_weights = [f.weight - extra_facet_layers_mean_weight for f in self.extra_facet_layers_for_target]

                for n in range(self.hparams.num_facets):
                    if self.add_extra_facet_layers_alternate:
                        pos_embedding[:, n, :] = torch.nn.functional.linear(pos_embedding[:, n, :], extra_facet_layers_for_target_alternate_weights[n], self.extra_facet_layers_for_target[n].bias)
                        neg_embedding[:, n, :] = torch.nn.functional.linear(neg_embedding[:, n, :], extra_facet_layers_for_target_alternate_weights[n], self.extra_facet_layers_for_target[n].bias)
                    else:
                        pos_embedding[:, n, :] = self.extra_facet_layers_for_target[n](pos_embedding[:, n, :])
                        neg_embedding[:, n, :] = self.extra_facet_layers_for_target[n](neg_embedding[:, n, :])

            elif len(self.extra_facet_layers) > 0:
                for n in range(self.hparams.num_facets):
                    if self.add_extra_facet_layers_alternate:
                        pos_embedding[:, n, :] = torch.nn.functional.linear(pos_embedding[:, n, :], extra_facet_layers_alternate_weights[n], self.extra_facet_layers[n].bias)
                        neg_embedding[:, n, :] = torch.nn.functional.linear(neg_embedding[:, n, :], extra_facet_layers_alternate_weights[n], self.extra_facet_layers[n].bias)
                    else:
                        pos_embedding[:, n, :] = self.extra_facet_layers[n](pos_embedding[:, n, :])
                        neg_embedding[:, n, :] = self.extra_facet_layers[n](neg_embedding[:, n, :])

            if self.hparams.add_extra_facet_nonlinearity:
                source_embedding = self.extra_facet_nonlinearity(source_embedding)
                pos_embedding = self.extra_facet_nonlinearity(pos_embedding)
                neg_embedding = self.extra_facet_nonlinearity(neg_embedding)

            # Normalize each facet embeddings for visualization purposes
            source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized = self._get_normalized_embeddings(source_embedding, pos_embedding, neg_embedding)

            self._calculate_loss_set_reg(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=True, is_before_extra=False)

            if self.hparams.num_facets > 1:
                self._calculate_facet_distances_mean(source_embedding_normalized, pos_embedding_normalized, neg_embedding_normalized, is_val=True, is_before_extra=False)

            if self.sum_into_single_embeddings is not None \
            and self.sum_into_single_embeddings in ("training_and_inference", "training_only"):
                source_embedding_summed = torch.sum(source_embedding, dim=1).unsqueeze(1)
                pos_embedding_summed = torch.sum(pos_embedding, dim=1).unsqueeze(1)
                neg_embedding_summed = torch.sum(neg_embedding, dim=1).unsqueeze(1)

            if self.use_target_token_embs_input:
                # Pass the source embeddings through extra scaling layers
                source_embedding_scaled = torch.clone(source_embedding)

                for n in range(self.hparams.num_facets):
                    source_embedding_scaled[:, n, :] = self.scaling_layers_for_input_embs[n](source_embedding_scaled[:, n, :])

                if self.use_target_token_embs_normalize:
                    source_embedding_scaled_normalized = torch.nn.functional.normalize(source_embedding_scaled, p=2, dim=-1)

        if self.use_multiple_losses:
            loss = 0

            loss_breakdowns = []

            for i, l in enumerate(self.loss_list):
                if 'use_target_token_embs' in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]['use_target_token_embs']:
                    if "use_target_token_embs_kmeans" in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]["use_target_token_embs_kmeans"]:
                        this_loss = l(source_embedding, pos_embedding_tokens_kmeans, neg_embedding_tokens_kmeans)
                    elif 'do_not_use_target_token_embs_mean' in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]['do_not_use_target_token_embs_mean']:
                        if self.use_target_token_embs_normalize:
                            if self.use_target_token_embs_input:
                                this_loss = l(source_embedding_scaled_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                            else:
                                this_loss = l(source_embedding_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                        else:
                            if self.use_target_token_embs_input:
                                this_loss = l(source_embedding_scaled, pos_embedding_tokens, neg_embedding_tokens)
                            else:
                                this_loss = l(source_embedding, pos_embedding_tokens, neg_embedding_tokens)
                    else:
                        this_loss = l(source_embedding, pos_embedding_tokens_mean, neg_embedding_tokens_mean)
                elif "sum_into_single_embeddings" in self.hparams.loss_config[i].keys() and self.hparams.loss_config[i]['sum_into_single_embeddings']:
                    this_loss = l(source_embedding_summed, pos_embedding_summed, neg_embedding_summed)
                else:
                    this_loss = l(source_embedding, pos_embedding, neg_embedding)

                self.log('val_loss_' + self.hparams.loss_config[i]["name"], this_loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, logger=True)

                loss_breakdowns.append(this_loss)

                loss = loss + self.hparams.loss_config[i]["weight"] * this_loss
        else:
            if "loss_use_target_token_embs" in self.hparams and self.hparams.loss_use_target_token_embs:
                if "loss_use_target_token_embs_kmeans" in self.hparams and self.hparams.loss_use_target_token_embs_kmeans:
                    loss = self.loss(source_embedding, pos_embedding_tokens_kmeans, neg_embedding_tokens_kmeans)
                elif "loss_do_not_use_target_token_embs_mean" in self.hparams and self.hparams.loss_do_not_use_target_token_embs_mean:
                    if self.use_target_token_embs_normalize:
                        if self.use_target_token_embs_input:
                            loss = self.loss(source_embedding_scaled_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                        else:
                            loss = self.loss(source_embedding_normalized, pos_embedding_tokens_normalized, neg_embedding_tokens_normalized)
                    else:
                        if self.use_target_token_embs_input:
                            loss = self.loss(source_embedding_scaled, pos_embedding_tokens, neg_embedding_tokens)
                        else:
                            loss = self.loss(source_embedding, pos_embedding_tokens, neg_embedding_tokens)
                else:
                    loss = self.loss(source_embedding, pos_embedding_tokens_mean, neg_embedding_tokens_mean)
            elif self.sum_into_single_embeddings is not None and self.sum_into_single_embeddings in ("training_and_inference", "training_only"):
                loss = self.loss(source_embedding_summed, pos_embedding_summed, neg_embedding_summed)
            else:
                loss = self.loss(source_embedding, pos_embedding, neg_embedding)

            self.log('val_loss_original', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        self.log('val_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        if self.use_multiple_losses:
            return {"loss": loss, "loss_breakdowns": loss_breakdowns}
        else:
            return {"loss": loss}

    def validation_epoch_end(self, outputs):
        # Refer to
        # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html
        # Note: Apparently what we get out of self.all_gather(outputs) here is a list, not a tensor
        avg_loss = torch.mean(torch.Tensor(self.all_gather([o["loss"] for o in outputs])))

        if self.trainer.is_global_zero:
            self.log('avg_val_loss', avg_loss, rank_zero_only=True, on_epoch=True, prog_bar=True)

        if self.use_multiple_losses:
            losses_all_batch = [[] for _ in range(len(self.hparams.loss_config))]

            for o in outputs:
                for i, l in enumerate(o["loss_breakdowns"]):
                    losses_all_batch[i].append(l)

            for i in range(len(self.hparams.loss_config)):
                avg_this_loss = torch.mean(torch.Tensor(self.all_gather(losses_all_batch[i])))

                if self.trainer.is_global_zero:
                    self.log('avg_val_loss_' + self.hparams.loss_config[i]["name"], avg_this_loss, rank_zero_only=True, on_epoch=True, prog_bar=True)
        else:
            self.log('avg_val_loss_original', avg_loss, rank_zero_only=True, on_epoch=True, prog_bar=True)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file')
    parser.add_argument('--val_file')

    parser.add_argument('--train_size', type=int)
    parser.add_argument('--val_size', type=int)

    parser.add_argument('--pretrained_model_name', default="allenai/scibert_scivocab_uncased", type=str)
    parser.add_argument('--model_behavior', default='quartermaster', choices=['quartermaster', 'specter'], type=str)
    parser.add_argument('--num_facets', default=1, type=int)
    parser.add_argument('--sum_into_single_embeddings', choices=['training_and_inference', 'training_only', 'inference_only'], type=str)

    parser.add_argument('--add_extra_facet_layers', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_for_target', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_alternate', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_after', nargs='*', type=int, help='Add extra facet layers right after the hidden states of specified encoder layers.')
    parser.add_argument('--init_bert_layer_facet_layers', default="default", choices=["default", "identity"], type=str)
    parser.add_argument('--add_bert_layer_facet_layers_alternate', default=False, action='store_true')

    parser.add_argument('--add_extra_facet_layers_initialize_with_nsp_weights', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_initialize_0_with_nsp_weights', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_initialize_with_identical_random_weights', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_layers_initialize_differently_for_target', default=False, action='store_true')
    parser.add_argument('--add_extra_facet_nonlinearity', default=False, action='store_true')

    parser.add_argument('--add_perturb_embeddings', default=False, action='store_true')

    parser.add_argument('--loss_config', type=json.loads)
    parser.add_argument('--loss_type', default='margin', choices=['margin', 'bce'], type=str)
    parser.add_argument('--loss_margin', default=1.0, type=float)
    parser.add_argument('--loss_distance', default='l2-norm', choices=['l2-norm', 'cosine', 'dot'], type=str)
    parser.add_argument('--loss_reduction', default='mean', choices=['mean', 'sum', 'none'], type=str)
    parser.add_argument('--loss_reduction_multifacet', default='mean', choices=['mean', 'min', 'max', 'logsumexp'], type=str)
    parser.add_argument('--loss_reduction_multifacet_target', default='mean', choices=['mean', 'min', 'max', 'logsumexp'], type=str)
    parser.add_argument('--loss_reduction_multifacet_query_first', default=False, action='store_true')

    parser.add_argument('--loss_use_target_token_embs', default=False, action='store_true')
    parser.add_argument('--loss_do_not_use_target_token_embs_mean', default=False, action='store_true')
    parser.add_argument('--loss_use_target_token_embs_kmeans', default=False, action='store_true')
    parser.add_argument('--loss_use_target_token_embs_input',  default=False, action='store_true')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--limit_val_batches', default=1.0, type=float) # Check 1.0 * 1 = 1 of the val set
    parser.add_argument('--val_check_interval', default=1.0, type=float) # 1.0 * 1 = Every 1 epoch
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int, help="kwarg passed to DataLoader")
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')

    parser.add_argument("--lr_scheduler",
                        default="linear",
                        choices=ARG_TO_SCHEDULER_CHOICES,
                        metavar=ARG_TO_SCHEDULER_METAVAR,
                        type=str,
                        help="Learning rate scheduler")

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_frac", default=0.1, type=float, help="Fraction of steps to perform linear warmup.")
    parser.add_argument("--adafactor", action="store_true")

    parser.add_argument('--log_every_n_steps', default=1, type=int)
    parser.add_argument('--flush_logs_every_n_steps', default=100, type=int)
    parser.add_argument('--wandb', default=False, action="store_true")

    parser.add_argument('--save_dir', required=True)

    parsed_args = parser.parse_args()

    if ',' in parsed_args.gpus:
        parsed_args.gpus = list(map(int, parsed_args.gpus.split(',')))
        parsed_args.total_gpus = len(parsed_args.gpus)
    else:
        parsed_args.gpus = int(parsed_args.gpus)
        parsed_args.total_gpus = parsed_args.gpus

    return parsed_args


def get_lightning_trainer_params(input_args):

    train_params = {}

    train_params["accumulate_grad_batches"] = input_args.grad_accum
    train_params['track_grad_norm'] = -1
    train_params['limit_val_batches'] = input_args.limit_val_batches
    train_params['val_check_interval'] = input_args.val_check_interval
    train_params['max_epochs'] = input_args.num_epochs
    train_params['reload_dataloaders_every_n_epochs'] = 1

    # PyTorch GPU related
    train_params["precision"] = 16 if input_args.fp16 else 32

    if (isinstance(input_args.gpus, int) and input_args.gpus > 1) or (isinstance(input_args.gpus, list) and len(input_args.gpus) > 1):
        train_params["accelerator"] = "ddp"
        # DDP optimizations
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        train_params["plugins"] = pl.plugins.DDPPlugin(
            find_unused_parameters=False,
            gradient_as_bucket_view=True)
    else:
        train_params["accelerator"] = None

    train_params['gpus'] = input_args.gpus
    train_params['amp_backend'] = 'native' # PyTorch AMP
    train_params['deterministic'] = True
    train_params['benchmark'] = False

    # log_every_n_steps how frequently pytorch lightning logs.
    # By default, Lightning logs every 50 rows, or 50 training steps.
    train_params['log_every_n_steps'] = input_args.log_every_n_steps
    train_params['flush_logs_every_n_steps'] = input_args.flush_logs_every_n_steps

    return train_params


if __name__ == '__main__':

    args = parse_args()

    # Create args.save_dir if it doesn't exist already
    pathlib.Path(args.save_dir).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(args.save_dir, 'logs')).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(args.save_dir, 'checkpoints')).mkdir(exist_ok=True)

    # Reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    pl.seed_everything(args.seed, workers=True)

    model = QuarterMaster(**vars(args))

    # logger used by trainer
    if args.wandb:
        pathlib.Path(os.path.join(args.save_dir, 'logs', 'wandb')).mkdir(exist_ok=True)

        pl_logger = pl.loggers.WandbLogger(
            name=args.save_dir,
            save_dir=os.path.join(args.save_dir, 'logs'))
    else:
        pl_logger = pl.loggers.TensorBoardLogger(
            name='pl-logs',
            save_dir=os.path.join(args.save_dir, 'logs'))

    pl_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.save_dir, 'checkpoints'),
        filename='ep-{epoch}_avg_val_loss-{avg_val_loss:.3f}',
        save_top_k=-1,
        save_last=1,
        every_n_val_epochs=1,
        verbose=True,
        monitor='avg_val_loss', # monitors metrics logged by self.log.
        mode='min')

    pl_learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    pl_other_trainer_params = get_lightning_trainer_params(args)

    trainer = pl.Trainer(
        logger=pl_logger,
        callbacks=[pl_checkpoint_callback, pl_learning_rate_callback],
        **pl_other_trainer_params)

    trainer.fit(model)
