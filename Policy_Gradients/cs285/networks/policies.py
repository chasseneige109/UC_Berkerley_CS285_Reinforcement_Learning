import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device) ## Method Chaining
            parameters = self.logits_net.parameters()
        else: ## continuous에선 Gaussian Policy
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device) ## Method Chaining
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            ) ## 표준편차 <- 얘도 학습 가능한 파라미터기 때문에 파라미터로 추가.
            parameters = itertools.chain([self.logstd], self.mean_net.parameters()) 
            ## chain : 리스트든 튜플이든 아무거나 다 받아서 쫙 펴서 썡원소들로 나열 

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:   ## -> 는 반환값 자료형 힌트
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action   # self, ob_dim, ac_dim, False, n_layer, layer_size, learning_rate
        obs_tensor = torch.FloatTensor(obs).to(ptu.device)
        
        action_distribution = self(obs_tensor)
        action = action_distribution.sample().cpu().detach().numpy()
        
        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            return distributions.Categorical(logits = logits)
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd) # std는 항상 양수인데, 신경망 파라미터는 음수도 가능하니 log 스케일로 저장해놓음. Normal에 넣을땐 다시 원래 스케일로 변환해야함.
            return distributions.Normal(mean,std)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs) # numpy인 obs를 tensor로 변환
        actions = ptu.from_numpy(actions) ## get_action으로 가져온, action_distribution에서 뽑은 실제 샘플
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        action_distribution = self(obs)
        log_prob = action_distribution.log_prob(actions) #action_distribution과 action은 같은 self(obs)에서 샘플링  했다는 사실을 우리가 알고 있기에 이렇게 사용가능.
        if not self.discrete:
            log_prob = log_prob.sum(axis=-1) 
        
        loss = -(log_prob * advantages).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
