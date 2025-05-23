import math
import torch
from torch.optim.optimizer import Optimizer
import time
import numpy as np
from config.configurator import configs

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class HMG(Optimizer):
    r"""Implements MetaBalance algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        relax factor: the hyper-parameter to control the magnitude proximity
        beta: the hyper-parameter to control the moving averages of magnitudes, set as 0.9 empirically

    """

    def __init__(self, params, relax_factor=0.7, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(HMG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array, nonshared_idx):  # , closure=None
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()
        self.balance_GradMagnitudes(loss_array, nonshared_idx)

        # return loss

    def balance_GradMagnitudes(self, loss_array, nonshared_idx):
        grad_task = []
        for loss_index, loss in enumerate(loss_array):
            loss.backward(retain_graph=True)
            for group in self.param_groups:
                for p_idx, p in enumerate(group['params']):
                    if loss_index == 0:
                        #if p.grad is not None:
                        grad_task.append(p.grad.detach().clone())

                    if p_idx == nonshared_idx:
                        continue

                    if p.grad is None:
                        #print("breaking")
                        break

                    if p.grad.is_sparse:
                        raise RuntimeError('HMG does not support sparse gradients')

                    # if p.grad.equal(torch.zeros_like(p.grad)):
                    #     continue

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        for j, _ in enumerate(loss_array):
                            if j == 0:
                                p.norms = [torch.zeros(1).to(configs['device'])]
                            else:
                                p.norms.append(torch.zeros(1).to(configs['device']))

                    # calculate moving averages of gradient magnitudes
                    beta = group['beta']
                    p.norms[loss_index] = (p.norms[loss_index] * beta) + ((1 - beta) * torch.norm(p.grad))

                    # narrow the magnitude gap between the main gradient and each auxilary gradient
                    relax_factor = group['relax_factor']

                    if p.norms[loss_index] > p.norms[0]:
                        inner_p = torch.sum(p.grad * grad_task[p_idx])
                        if inner_p < 0:
                            p.grad = p.grad - inner_p / (torch.norm(grad_task[p_idx]) ** 2) * grad_task[p_idx]
                        p.grad = (p.norms[0] * p.grad / p.norms[loss_index]) * relax_factor + p.grad * (
                                1.0 - relax_factor)

                    if loss_index == 0:
                        state['sum_gradient'] = torch.zeros_like(p.data)
                        state['sum_gradient'] += p.grad
                    else:
                        state['sum_gradient'] += p.grad

                    # have to empty p.grad, otherwise the gradient will be accumulated
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                    if loss_index == len(loss_array) - 1:
                        p.grad = state['sum_gradient']