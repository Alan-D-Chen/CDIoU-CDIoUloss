# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
from numpy import *
import time
import torch.nn.functional as F

from .bounding_box import BoxList

from atss_core.layers import nms as _box_nms
from atss_core.layers import ml_nms as _box_ml_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def boxlist_ml_nms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)
    keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
# IoU for CD IoU
def box_diou(blist1, blist2):
    # print("!!!!", type(blist1[1]))
    # a = torch.Tensor([blist1[0], blist1[1]]).view(1, 2)
    # b = torch.Tensor([blist2[0], blist2[1]]).view(1, 2)
    dist = F.pairwise_distance(torch.Tensor([blist1[0], blist1[1]]).view(1, 2),
                               torch.Tensor([blist2[0], blist2[1]]).view(1, 2), p=2) + \
           F.pairwise_distance(torch.Tensor([blist1[2], blist1[3]]).view(1, 2),
                               torch.Tensor([blist2[2], blist2[3]]).view(1, 2), p=2)
           # F.pairwise_distance(torch.Tensor([blist1[0], blist1[3]]).view(1, 2),
           #                     torch.Tensor([blist2[0], blist2[3]]).view(1, 2), p=2) + \
           # F.pairwise_distance(torch.Tensor([blist1[1], blist1[2]]).view(1, 2),
           #                     torch.Tensor([blist2[1], blist2[2]]).view(1, 2), p=2)

    # print("<------->", dist)
    x1_min = min(blist1[0], blist2[0])
    y1_min = min(blist1[1], blist2[1])
    x2_max = max(blist1[2], blist2[2])
    y2_max = max(blist1[3], blist2[3])
    dist2 = F.pairwise_distance(torch.Tensor([x1_min, y1_min]).view(1, 2),
                               torch.Tensor([x2_max, y2_max]).view(1, 2), p=2)
    dist = dist/(2*dist2)
    return dist

def box_dious(blist1, blist2):
    # Get the minimum bounding box (including region proprosal and ground truth) #
    # n = torch.rand([3, 4])
    # m = torch.rand([5, 4])
    n = blist1.cuda()
    m = blist2.cuda()

    #print("n.shape[0],n.shape[1]:", n.shape[0], n.shape[1])
    #print("m.shape[0],m.shape[1]:", m.shape[0], m.shape[1])
    nd0 = n.shape[0]
    md0 = m.shape[0]
    #print("##################################################################-------->>")
    nss = n.unsqueeze(1)
    nss = nss.expand(nd0, md0, 4)
    #print("n & n.shape:\n", n, "\n", n.shape)
    mss = m.unsqueeze(0)
    mss = mss.expand(nd0, md0, 4)
    #print("m & m.shape:\n", m, "\n", m.shape)
    nms = torch.cat((nss, mss), dim=2)
    #print("nms & nms.shape:\n", nms, "\n", nms.shape)

    A = nms[:, :, [0, 4]]
    B = nms[:, :, [1, 5]]
    C = nms[:, :, [2, 6]]
    D = nms[:, :, [3, 7]]
    #print("#########################")
    #print("A & A.shape:\n", A, "\n", A.shape)
    Am = torch.max(A, 2)[0]
    #print("Am & Am.shape:\n", Am, "\n", Am.shape)
    #print(B, B.shape)
    Bm = torch.max(B, 2)[0]
    #print(Bm)
    #print(C, C.shape)
    Cm = torch.max(C, 2)[0]
    #print(Cm)
    #print(D, D.shape)
    Dm = torch.max(D, 2)[0]
    #print(Dm)
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    AB = torch.cat((Am, Bm), dim=1).cuda()
    CD = torch.cat((Cm, Dm), dim=1).cuda()
    #print("AB & AB.shape:\n", AB, "\n", AB.shape)
    #print("CD & CD.shape:\n", CD, "\n", CD.shape)
    XY = torch.zeros([Am.shape[0], Am.shape[1], 4]).cuda()
    #print("XY & XY.shape:\n", XY, "\n", XY.shape)
    XY[:, :, 0] = Am
    XY[:, :, 1] = Bm
    XY[:, :, 2] = Cm
    XY[:, :, 3] = Dm
    #print("XY & XY.shape:\n", XY, "\n", XY.shape)
    XYx = (XY[:, :, [2, 3]] - XY[:, :, [0, 1]]) ** 2
    #print("XYx & XYx.shape:\n", XYx, "\n", XYx.shape)
    XxY = XYx[:, :, 0] + XYx[:, :, 1]
    XYs = XxY.sqrt()  ###########################-> to get square root
    #print("XYs & XYs.shape:\n", XYs, "\n", XYs.shape)

    #######################################################
    #######################################################
    # The average distance between GT and RP is obtained #
    nms = torch.cat((n, m), dim=0)
    #print("nms & nms.shape:\n", nms, "\n", nms.shape)
    #########################################################
    #print("n.shape & n:\n", n.shape, "\n", n)
    n0 = n[:, [0, 3]]  # .unsqueeze(1)
    n1 = n[:, [1, 2]]  # .unsqueeze(1)
    #print("n0 & n0.shape:\n", n0, "\n", n0.shape)
    #print("n1 & n1.shape:\n", n1, "\n", n1.shape)
    ns = torch.cat((n, n0, n1), dim=1)
    #print("ns.shape & ns:\n", ns.shape, "\n", ns)
    ######################################################
    #print("m.shape & m:\n", m.shape, "\n", m)
    m0 = m[:, [0, 3]]  # .unsqueeze(1)
    m1 = m[:, [1, 2]]  # .unsqueeze(1)
    #print("m0 & m0.shape:\n", m0, "\n", m0.shape)
    #print("m1 & m1.shape:\n", m1, "\n", m1.shape)
    ms = torch.cat((m, m0, m1), dim=1)
    #print("ms.shape & ms:\n", ms.shape, "\n", ms)
    ################################################################
    ns = ns.unsqueeze(1)
    ms = ms.unsqueeze(0)
    #print("ns.shape & ns->unsqueeze:\n", ns.shape, "\n", ns)
    #print("ms.shape & ms->unsqueeze:\n", ms.shape, "\n", ms)

    n = ns
    m = ms
    #print("n.shape & n:\n", n.shape, "\n", n)
    #print("m.shape & m:\n", m.shape, "\n", m)
    tmp = (n - m) ** 2
    #print("tmp.shape:\n", tmp.shape)
    #print("tmp1->(n - m) ** 2:\n", tmp)
    # tmp = tmp[0:-1:2] + tmp[1:-1:2]
    # print(tmp[:,:,0::2],"\n", tmp[:,:,1::2])
    tmps = tmp[:, :, 0::2] + tmp[:, :, 1::2]
    #print("tmps and tmps.shape:\n", tmps, "\n", tmps.shape)
    # tmp = np.sqrt(tmps)               #######################-> to get square root
    tmp = tmps.sqrt()
    #print("tmp3->tmps-square root:\n", tmp, "\n", tmp.shape)
    # tmp = tmp.mean(axis=2, keepdim=False)/4
    tmp = torch.mean(tmp, dim=2, keepdim=False) / 4
    #print("tmp->mean:\n", tmp, "\n", tmp.shape)

    # get DIoU+ #
    #print("DIoU+ and DIoU.shape:\n", tmp / XYs, "\n", (tmp / XYs).shape)
    tmpx = (tmp / XYs).cuda()
    return (tmpx)

def boxlist_ioux(boxlist1, boxlist2):
    time_start = time.time()
    """
    Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    # information for boxlist1 and boxlist2:
    """
    print("##########################################################################")
    print(" length of boxlist1:", len(boxlist1))
    print(" type of boxlist1:", type(boxlist1))
    print(" boxlist1:\n", boxlist1)

    print(" length of boxlist2:", len(boxlist2))
    print(" type of boxlist2:", type(boxlist2))
    print(" boxlist2:\n", boxlist2)
    """
    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox
    """
    print(" shape of box1:",box1.shape)
    print(" type of box1:", type(box1))
    print(" box1:\n", box1)

    print(" shape of box2:", box2.shape)
    print(" type of box2:", type(box2))
    print(" box2:\n", box2)
    """

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    """
    print(" shape of iou:", iou.shape)
    print(" type of iou:", type(iou))
    print(" iou:\n", iou)
    """
    # Calculation for DIoU #
    ## new way to get diou+
    diou = box_dious(box1, box2)
    dious = diou
    diou = 0.0001 * (1 - diou)

    iou = iou + diou
    """
    print(" shape of diou:", diou.shape)
    print(" type of diou:", type(diou))
    print(" diou:\n", diou)

    print(" shape of iou:", iou.shape)
    print(" type of iou:", type(iou))
    print(" iou:\n", iou)

    print("##########################################################################")
    """
    """
    improvement for IoU ,-->CD-IoU,go to boxlist_ious(boxlist1,boxlist2):
    """
    return iou, dious
## JUST for fine-training in lab #501 with TeslaV100 and 2 GPUs ## Training
def boxlist_iou(boxlist1, boxlist2):
    time_start = time.time()
    """
    Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    # information for boxlist1 and boxlist2:
    """
    print("##########################################################################")
    print(" length of boxlist1:", len(boxlist1))
    print(" type of boxlist1:", type(boxlist1))
    print(" boxlist1:\n", boxlist1)

    print(" length of boxlist2:", len(boxlist2))
    print(" type of boxlist2:", type(boxlist2))
    print(" boxlist2:\n", boxlist2)
    """
    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox
    """
    print(" shape of box1:",box1.shape)
    print(" type of box1:", type(box1))
    print(" box1:\n", box1)

    print(" shape of box2:", box2.shape)
    print(" type of box2:", type(box2))
    print(" box2:\n", box2)
    """

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    """
    print(" shape of iou:", iou.shape)
    print(" type of iou:", type(iou))
    print(" iou:\n", iou)
    """
    # Calculation for DIoU #
    ## new way to get diou+
    diou = box_dious(box1, box2)
    dious = diou
    diou = 0.001 * (1 - diou)
    iou = iou + diou
    """
    improvement for IoU -->CD-IoU,go to boxlist_ious(boxlist1,boxlist2):
    """
    #time_end = time.time()
    #print('ATSS/atss_core/structures/boxlist_ops.py and time cost is ', time_end - time_start, 's')
    return iou
## JUST for Original configuration Training
def boxlist_ious(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)

    return iou

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    cat => concatnate
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    # This series of assert embodies the rigor #
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
