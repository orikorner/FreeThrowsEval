from moderator.moderators import ClassifierModerator, TrajectoryModerator


def get_training_moderator(config, net, lr, trj_weight=0.5):
    if config.objective_mode == 'trj':
        return TrajectoryModerator(config, net, lr, trj_weight)
    else:
        return ClassifierModerator(config, net, lr)
