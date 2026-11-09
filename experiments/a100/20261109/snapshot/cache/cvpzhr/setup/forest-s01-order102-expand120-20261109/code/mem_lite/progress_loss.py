"""Class-distribution supervision plus symmetric continuous-progress error."""


def progress_objective(per_sample_ce, prediction, target, coefficient):
    return per_sample_ce + coefficient * (prediction.float()-target.float()).square()
