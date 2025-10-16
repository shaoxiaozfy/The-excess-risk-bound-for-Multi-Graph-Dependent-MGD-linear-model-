from models.deep_model import *
from models.lightxml import LightXML


def get_model_by_name(args, label_map, total_class_num=None):
    if args.model_name == "VGG11":
        model = VGG11(num_classes=total_class_num)
    elif args.model_name == "ResNet34":
        model = ResNet34(num_classes=total_class_num)
    elif args.model_name == "SimpleCNN":
        model = SimpleCNN(num_classes=total_class_num)
    elif args.model_name == "ResNet18":
        model = ResNet18(num_classes=total_class_num)
    elif args.model_name == "ResNet50":
        model = ResNet50(num_classes=total_class_num)
    elif args.model_name == "ResNet101":
        model = ResNet101(num_classes=total_class_num)
    elif args.model_name == "lm":
        model = LightXML(n_labels=len(label_map), bert=args.bert,
                         update_count=args.update_count,
                         use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                         loss_name=args.loss_name)
    else:
        raise ValueError("unknown model type")
    return model
