def forward_trace(self, img):
    """
    trace network.
    """
    assert self.with_bbox, 'Bbox head must be implemented.'
    x = self.extract_feat(img)
    rpn_outs = self.rpn_head(x)

    cls_all = list()
    bbox_all = list()

    for (cls, bbox) in zip(rpn_outs[0], rpn_outs[1]):
        cls_all.append(cls.permute(0, 2, 3, 1).contiguous())
        bbox_all.append(bbox.permute(0, 2, 3, 1).contiguous())

    loc = torch.cat([o.view(o.size(0), -1) for o in cls_all], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in bbox_all], 1)

    output = torch.cat((conf.view(conf.size(0), -1, 4), loc.view(loc.size(0), -1, self.rpn_head.cls_out_channels)),
                       2)
    output = output.view(-1, 1)
    data = torch.cat([xx.reshape(-1, 1) for xx in x], 0)
    output = torch.cat([output, data], 0)

    print(output.shape)
    return output
