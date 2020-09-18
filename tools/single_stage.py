def forward_trace(self, img):
    """
    trace network.
    """
    x = self.extract_feat(img)
    outs = self.bbox_head(x)

    cls_all = list()
    bbox_all = list()
    for (cls, bbox) in zip(outs[0], outs[1]):
        cls_all.append(cls.permute(0, 2, 3, 1).contiguous())
        bbox_all.append(bbox.permute(0, 2, 3, 1).contiguous())

    loc = torch.cat([o.view(o.size(0), -1) for o in cls_all], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in bbox_all], 1)

    output = torch.cat((conf.view(conf.size(0), -1, 4), loc.view(loc.size(0), -1, self.bbox_head.cls_out_channels)), 2)

    print(output.shape)
    return output
