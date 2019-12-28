from collections import namedtuple

Game = namedtuple('Game', [
    'env_name', 'time_factor', 'input_size', 'output_size', 'layers',
    'activation', 'noise_bias', 'output_noise', 'rnn_mode', 'experimental_mode'
])

games = {}

vae_racing = Game(
    env_name='VAERacing-v0',
    input_size=16,
    output_size=3,
    time_factor=0,
    layers=[10, 0],
    activation='tanh',
    noise_bias=0.0,
    output_noise=[False, False, False],
    rnn_mode=False,
    experimental_mode=False,
)
games['vae_racing'] = vae_racing

learn_vae_racing = Game(
    env_name='VAERacing-v0',
    input_size=16,
    output_size=3,
    time_factor=0,
    layers=[10, 10],
    activation='tanh',
    noise_bias=0.0,
    output_noise=[False, False, False],
    rnn_mode=False,
    experimental_mode=True,
)
games['learn_vae_racing'] = learn_vae_racing
