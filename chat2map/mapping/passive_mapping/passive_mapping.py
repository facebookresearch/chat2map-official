# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class PassiveMapping(nn.Module):
    def __init__(
        self,
        actor_critic,
    ):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts):
        raise NotImplementedError

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass
