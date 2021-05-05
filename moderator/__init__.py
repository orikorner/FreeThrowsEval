from moderator.moderators import Moderator


def get_training_moderator(config, net, lr):
    return Moderator(config, net, lr)
