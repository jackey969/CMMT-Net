import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss

def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss

def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))

def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class softDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(softDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), "predict & target shape do not match"
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VAT2d_v2_New_Data(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1, num_classes=4):
        super(VAT2d_v2_New_Data, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(num_classes)

    def forward(self, model, x):
        with torch.no_grad():
            output = model(x)
            pred = [F.softmax(output[i], dim=1) for i in range(len(output))]

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)
                logp_hat = [F.softmax(pred_hat[i], dim=1) for i in range(len(pred_hat))]
                adv_distance = 0
                for i in range(len(pred)):
                    for j in range(len(pred)):
                        if i != j:
                            adv_distance += self.loss(logp_hat[i], pred[j])
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)
            logp_hat = [F.softmax(pred_hat[i], dim=1) for i in range(len(pred_hat))]
            lds = 0
            for i in range(len(pred)):
                for j in range(len(pred)):
                    if i != j:
                        lds += self.loss(logp_hat[i], pred[j])
        return lds


class VAT2d_v2_MT(nn.Module):
    def __init__(self, xi=10.0, epi=6.0, ip=1, num_classes=4):
        super(VAT2d_v2_MT, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = softDiceLoss(num_classes)

    def forward(self, model, ema_model, x):
        with torch.no_grad():
            ema_output = ema_model(x)
            ema_pred = [F.softmax(ema_output[i], dim=1) for i in range(len(ema_output))]

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)
                logp_hat = [F.softmax(pred_hat[i], dim=1) for i in range(len(pred_hat))]
                adv_distance = 0
                for i in range(len(pred_hat)):
                    for j in range(len(pred_hat)):
                        if i != j:
                            adv_distance += self.loss(logp_hat[i], ema_pred[j])
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            pred_hat = model(x + r_adv)
            logp_hat = [F.softmax(pred_hat[i], dim=1) for i in range(len(pred_hat))]
            lds = 0
            for i in range(len(pred_hat)):
                for j in range(len(pred_hat)):
                    if i != j:
                        lds += self.loss(logp_hat[i], ema_pred[j])
        return lds
