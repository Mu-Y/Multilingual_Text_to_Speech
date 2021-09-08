# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import quadprog

from .common import MLP, ResNet18

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class GEM(nn.Module):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 mem_data,
                 n_tasks,
                 hp):
        """
        mem_data: data loader of buffered prev samples
        n_tasks: num of total tasks. should include current task. e.g when training french
                 n_tasks should be 2.
        """
        super(Net, self).__init__()
        self.hp = hp
        # nl, nh = args.n_layers, args.n_hiddens
        self.margin = hp.memory_strength
        self.n_tasks = n_tasks
        # self.is_cifar = (args.data_file == 'cifar100.pt')
        # if self.is_cifar:
        #     self.net = ResNet18(n_outputs)
        # else:
        #     self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.net = model
        self.criterion = criterion

        # self.ce = nn.CrossEntropyLoss()
        # self.n_outputs = n_outputs

        # self.opt = optim.SGD(self.parameters(), args.lr)
        self.opt = optimizer

        # self.n_memories = args.n_memories
        # self.gpu = args.cuda

        # allocate episodic memory
        self.mem_data = mem_data
        # self.memory_data = torch.FloatTensor(
        #     n_tasks, self.n_memories, n_inputs)
        # self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        # if args.cuda:
        #     self.memory_data = self.memory_data.cuda()
        #     self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks).cuda()
        # if args.cuda:
        #     self.grads = self.grads.cuda()

        # # allocate counters
        # self.observed_tasks = []
        # self.old_task = -1
        # self.mem_cnt = 0
        # if self.is_cifar:
        #     self.nc_per_task = int(n_outputs / n_tasks)
        # else:
        #     self.nc_per_task = n_outputs

    def forward(self, x, t):
        # output = self.net(x)
        # parse batch
        # batch = list(map(to_gpu, x))
        src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch

        # get teacher forcing ratio
        if self.hp.constant_teacher_forcing: tf = self.hp.teacher_forcing
        else: tf = cos_decay(max(global_step - self.hp.teacher_forcing_start_steps, 0), self.hp.teacher_forcing_steps)

        # run the current model (student)
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = self.model(src, src_len, trg_mel, trg_len, spkrs, langs, tf)
        # if self.is_cifar:
        #     # make sure we predict classes within the current task
        #     offset1 = int(t * self.nc_per_task)
        #     offset2 = int((t + 1) * self.nc_per_task)
        #     if offset1 > 0:
        #         output[:, :offset1].data.fill_(-10e10)
        #     if offset2 < self.n_outputs:
        #         output[:, offset2:self.n_outputs].data.fill_(-10e10)
        # return output
        return post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output

    def parse_batch_by_task(self, x, sel_indices):
        """
        select samples from a batch that correspond to a task
        """
        if x is None: return x
        return torch.index_select(x, 0, sel_indices)


    def observe(self, cur_batch, cur_task):
        # # update memory
        # if t != self.old_task:
        #     self.observed_tasks.append(t)
        #     self.old_task = t

        # # Update ring buffer storing examples from current task
        # bsz = y.data.size(0)
        # endcnt = min(self.mem_cnt + bsz, self.n_memories)
        # effbsz = endcnt - self.mem_cnt
        # self.memory_data[t, self.mem_cnt: endcnt].copy_(
        #     x.data[: effbsz])
        # if bsz == 1:
        #     self.memory_labs[t, self.mem_cnt] = y.data[0]
        # else:
        #     self.memory_labs[t, self.mem_cnt: endcnt].copy_(
        #         y.data[: effbsz])
        # self.mem_cnt += effbsz
        # if self.mem_cnt == self.n_memories:
        #     self.mem_cnt = 0

        # # compute gradient on previous tasks
        # if len(self.observed_tasks) > 1:
        #     for tt in range(len(self.observed_tasks) - 1):
        #         self.zero_grad()
        #         # fwd/bwd on the examples in the memory
        #         past_task = self.observed_tasks[tt]

        #         offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
        #                                            self.is_cifar)
        #         ptloss = self.ce(
        #             self.forward(
        #                 self.memory_data[past_task],
        #                 past_task)[:, offset1: offset2],
        #             self.memory_labs[past_task] - offset1)
        #         ptloss.backward()
        #         store_grad(self.parameters, self.grads, self.grad_dims,
        #                    past_task)

        # compute gradient on previous tasks
        for task_idx in range(self.n_tasks - 1):
            # -1 means exclude current task
            self.zero_grad()
            for batch in self.mem_data:
                batch = list(map(to_gpu, batch))
                # src, src_len, trg_mel, trg_lin, trg_len, stop_trg, spkrs, langs = batch
                bs = batch[0].size(0)
                batch_langs = batch[7]
                # task_idx: an int representing task identifier.
                # e.g. 0 for german, 1 for french, 2 for spanish
                sel_indices = torch.cuda.LongTensor(
                         [i for i in range(bs) if batch_langs[i][0][task_idx]==1])
                batch_task = [self.parse_batch_by_task(x, 0, sel_indices) for x in batch]
                ### TODO make sure batch_langs will not be affected by the line above
                pdb.set_trace()
                post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = self.forward(batch_task)
                # evaluate loss function
                # post_trg = trg_lin if self.hp.predict_linear else trg_mel
                # classifier = self.model._reversal_classifier if self.hp.reversal_classifier else None
                ptloss, _ = self.criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, trg_mel, stop_pred, stop_trg, alignment, spkrs, spkrs_pred, enc_output, None)
                ptloss.backward()
            store_grad(self.parameters, self.grads, self.grad_dims, task_idx)


        # now compute the grad on the current minibatch
        self.zero_grad()

        # offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        # loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        batch = list(map(to_gpu, cur_batch))
        post_pred, pre_pred, stop_pred, alignment, spkrs_pred, enc_output = self.forward(batch)
        # post_trg = trg_lin if self.hp.predict_linear else trg_mel
        # classifier = self.model._reversal_classifier if self.hp.reversal_classifier else None
        loss, _ = self.criterion(src_len, trg_len, pre_pred, trg_mel, post_pred, trg_mel, stop_pred, stop_trg, alignment, spkrs, spkrs_pred, enc_output, None)
        loss.backward()

        # # check if gradient violates constraints
        # if len(self.observed_tasks) > 1:
        #     # copy gradient
        #     store_grad(self.parameters, self.grads, self.grad_dims, t)
        #     indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
        #         else torch.LongTensor(self.observed_tasks[:-1])
        #     dotp = torch.mm(self.grads[:, t].unsqueeze(0),
        #                     self.grads.index_select(1, indx))
        #     if (dotp < 0).sum() != 0:
        #         project2cone2(self.grads[:, t].unsqueeze(1),
        #                       self.grads.index_select(1, indx), self.margin)
        #         # copy gradients back
        #         overwrite_grad(self.parameters, self.grads[:, t],
        #                        self.grad_dims)
        # self.opt.step()

        # check if gradient violates constraints
        # copy gradient
        store_grad(self.parameters, self.grads, self.grad_dims, cur_task)
        indx = torch.cuda.LongTensor(torch.arange(cur_task))
        dotp = torch.mm(self.grads[:, cur_task].unsqueeze(0),
                        self.grads.index_select(1, indx))
        if (dotp < 0).sum() != 0:
            project2cone2(self.grads[:, cur_task].unsqueeze(1),
                          self.grads.index_select(1, indx), self.margin)
            # copy gradients back
            overwrite_grad(self.parameters, self.grads[:, cur_task],
                           self.grad_dims)
        self.opt.step()
