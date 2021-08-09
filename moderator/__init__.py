from moderator.moderators import ClassifierModerator, TrajectoryModerator


def get_training_moderator(config, net, lr):
    if config.in_pretrain:
        return TrajectoryModerator(config, net, lr)
    else:
        return ClassifierModerator(config, net, lr)
