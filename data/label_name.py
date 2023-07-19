
def get_label_name(cfg, dataset):
    if dataset == "cifar10":
        return cifar10_label()
    elif dataset == "cifar100":
        pass


def cifar10_label():
    label_name = ["an image of airplane",
                      "an image of automobile",
                      "an image of bird",
                      "an image of cat",
                      "an image of deer",
                      "an image of dog",
                      "an image of frog",
                      "an image of horse",
                      "an image of ship",
                      "an image of truck"
                      ]
    return label_name