from utils.register import register


def load_model(input_dim: int, output_dim: int, config):
    return register.models[config.model](input_dim=input_dim, output_dim=output_dim, **vars(config))