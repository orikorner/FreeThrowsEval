from moderator.moderators import Moderator


def get_training_moderator(config, net):
    return Moderator(config, net)
