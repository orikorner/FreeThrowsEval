from moderator.moderators import ClassifierModerator, TrajectoryModerator


def get_training_moderator(config, net, lr):
    if config.objective_mode == 'trj':
        return TrajectoryModerator(config, net, lr)
    else:
        return ClassifierModerator(config, net, lr)
