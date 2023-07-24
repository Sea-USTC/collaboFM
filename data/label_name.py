
def get_label_name(cfg, dataset):
    if dataset == "cifar10":
        return cifar10_label()
    elif dataset == "cifar100":
        pass


def cifar10_label():
    label_name = ["a photo of airplane",
                  "a photo of automobile",
                  "a photo of bird",
                  "a photo of cat",
                  "a photo of deer",
                  "a photo of dog",
                  "a photo of frog",
                  "a photo of horse",
                  "a photo of ship",
                  "a photo of truck"
                ]
    return label_name