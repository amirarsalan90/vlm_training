def freeze_network(model):
    for param in model.language_model.parameters():
        param.requires_grad = False
    
    for param in model.vision_tower.parameters():
        param.requires_grad = False


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)