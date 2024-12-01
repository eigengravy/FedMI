import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np


def _mask(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).bool()


def _regularized_loss(mi, reg):
    loss = mi - reg
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss


def _mine(logits, labels):
    batch_size, classes = logits.shape
    joints = torch.masked_select(logits, _mask(labels, classes))
    t = torch.mean(joints)
    et = torch.logsumexp(logits, dim=(0, 1)) - np.log(batch_size * classes)
    return t, et, joints, logits.flatten()


def _infonce(logits, labels):
    _, classes = logits.shape
    joints = torch.masked_select(logits, _mask(labels, classes))
    t = joints.mean()
    et = logits.logsumexp(dim=1).mean() - np.log(classes)
    return t, et, joints, logits.flatten()


def _smile(logits, labels, clip):
    t, et, _, _ = _mine(torch.clamp(logits, -clip, clip), labels)
    _, _, joints, marginals = _mine(logits, labels)
    return t, et, joints, marginals


def _tuba(logits, labels, clip, a_y):
    _, classes = logits.shape
    joint = torch.masked_select(logits, _mask(labels, classes))
    marginal = logits.flatten()
    if clip > 0.0:
        t = torch.clip(joint, -clip, clip).mean()
        et = torch.clip(marginal, -clip, clip).exp().mean() / a_y + np.log(a_y) - 1.0
    else:
        t = joint.mean()
        et = marginal.exp().mean() / a_y + np.log(a_y) - 1.0

    return t, et, joint, marginal


def _js(logits, labels):
    _, classes = logits.shape
    joints = torch.masked_select(logits, _mask(labels, classes))
    marginals = logits.flatten()
    t = -torch.nn.functional.softplus(-joints).mean()
    et = torch.nn.functional.softplus(marginals).mean()
    return t, et, joints, marginals


def mine(logits, labels):
    t, et, joints, marginals = _mine(logits, labels)
    return t - et, joints, marginals


def mine(logits, labels):
    t, et, joint, marginal = _mine(logits, labels)
    return t - et, joint, marginal


def remine(logits, labels, alpha, bias):
    t, et, joints, marginals = _mine(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joints, marginals


def infonce(logits, labels):
    t, et, joint, marginal = _infonce(logits, labels)
    return t - et, joint, marginal


def reinfonce(logits, labels, alpha, bias):
    t, et, joint, marginal = _infonce(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def smile(logits, labels, clip):
    t, et, joint, marginal = _smile(logits, labels, clip)
    return t - et, joint, marginal


def resmile(logits, labels, clip, alpha, bias):
    t, et, joint, marginal = _smile(logits, labels, clip)
    _, reg_et, _, _ = _mine(logits, labels)
    reg = alpha * torch.square(reg_et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def tuba(logits, labels):
    t, et, joint, marginal = _tuba(logits, labels, 0.0, 1.0)
    return t - et, joint, marginal


def nwj(logits, labels):
    t, et, joint, marginal = _tuba(logits, labels, 0.0, np.e)
    return t - et, joint, marginal


def retuba(logits, labels, clip, alpha):
    t, et, joint, marginal = _tuba(logits, labels, clip, 1.0)
    _, _, _, reg_marginal = _tuba(logits, labels, 0.0, 1.0)
    reg = alpha * torch.square(
        torch.logsumexp(reg_marginal, dim=0) - np.log(reg_marginal.shape[0])
    )
    return _regularized_loss(t - et, reg), joint, marginal


def renwj(logits, labels, clip, alpha):
    t, et, joint, marginal = _tuba(logits, labels, clip, np.e)
    _, _, _, reg_marginal = _tuba(logits, labels, 0.0, np.e)
    reg = alpha * torch.square(
        torch.logsumexp(reg_marginal, dim=0) - np.log(reg_marginal.shape[0])
    )
    return _regularized_loss(t - et, reg), joint, marginal


def js(logits, labels):
    t, et, joint, marginal = _js(logits, labels)
    return t - et, joint, marginal


def rejs(logits, labels, alpha, bias):
    t, et, joint, marginal = _js(logits, labels)
    reg = alpha * torch.square(et - bias)
    return _regularized_loss(t - et, reg), joint, marginal


def nwjjs(logits, labels):
    loss, joint, marginal = js(logits, labels)
    mi, _, _ = nwj(logits, labels)
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss, joint, marginal


def renwjjs(logits, labels, alpha, bias, clip):
    loss, joint, marginal = rejs(logits, labels, alpha, bias)
    mi, _, _ = nwj(logits, labels)
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss, joint, marginal


def dmi(output, target, num_classes):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), num_classes).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

criterions = {
    "ce": nn.functional.cross_entropy,
    "mine": mine,
    "infonce": infonce,
    "smile_t1": partial(smile, clip=1.0),
    "smile_t10": partial(smile, clip=10.0),
    "tuba": tuba,
    "nwj": nwj,
    "js": js,
    "nwjjs": nwjjs,
    "dmi": dmi
}


for alpha in (0.1, 0.01, 0.001):
    criterions[f"remine_a{alpha}_b0"] = partial(remine, alpha=alpha, bias=0)
    criterions[f"reinfonce_a{alpha}_b0"] = partial(reinfonce, alpha=alpha, bias=0)
    criterions[f"resmile_t10_a{alpha}_b0"] = partial(
        resmile, clip=10, alpha=alpha, bias=0
    )
    criterions[f"renwj_t10_a{alpha}"] = partial(renwj, clip=10, alpha=alpha)
    criterions[f"retuba_t10_a{alpha}"] = partial(retuba, clip=10, alpha=alpha)
    criterions[f"rejs_a{alpha}_b1"] = partial(rejs, alpha=alpha, bias=1.0)
    criterions[f"renwjjs_a{alpha}_b1"] = partial(
        renwjjs, alpha=alpha, bias=1.0, clip=0.0
    )
