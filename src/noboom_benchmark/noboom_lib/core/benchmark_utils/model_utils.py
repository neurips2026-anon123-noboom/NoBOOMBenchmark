"""Helpers for registering configuration argument links per model."""

from collections import defaultdict
from typing import Dict, List, Tuple

_ARG_REGISTRY: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

def register_args(model: str, from_name: str, to_name: str, apply_on: str = 'instantiate'):
    """Record an argument mapping for a specific model.

    Args:
        model (str): Model name to register.
        from_name (str): Source argument name.
        to_name (str): Target argument name.
        apply_on (str): When to apply the mapping. Defaults to "instantiate".

    Returns:
        None: The registry is updated in place.
    """
    _ARG_REGISTRY[model].append((from_name, to_name, apply_on))

def get_args(model: str) -> list[Tuple[str, str, str]]:
    """Return registered argument mappings for the given model name.

    Args:
        model (str): Model name to query.

    Returns:
        list[Tuple[str, str, str]]: List of argument mappings.
    """
    return _ARG_REGISTRY.get(model, [])


# Registration
register_args("lstm_ae", "data.num_features", "model.network.init_args.input_dimension")
register_args("lstm_p", "data.num_features", "model.network.init_args.input_dim")
register_args("gdn", "data.num_features", "model.network.init_args.node_num")
register_args("gmm_vae", "data.num_features", "model.network.init_args.input_dim")
register_args("fedformer", "data.num_features", "model.network.init_args.input_dim")
register_args("timesnet", "data.num_features", "model.network.init_args.input_dim")
register_args("hpad", "data.num_features", "model.network.init_args.input_dim")
register_args("neutralad", "data.num_features", "model.network.init_args.ts_channels")
register_args("physdiff", "data.num_features", "model.network.init_args.num_channels")
register_args("gdn", "data.num_features", "model.detector.init_args.num_features")
register_args("lstm_ae", "data.num_features", "model.detector.init_args.num_features")
register_args("lstm_p", "data.num_features", "model.detector.init_args.num_features")
register_args("energyad_autoformer", "data.num_features", "model.network.init_args.input_dim")
register_args("energyad_fedformer", "data.num_features", "model.network.init_args.input_dim")
register_args("energyad_filonovlstm", "data.num_features", "model.network.init_args.input_dim")
register_args("paano", "data.num_features", "model.network.init_args.input_dim")

register_args("neutralad", "model.losses", "model.detector.init_args.loss", apply_on='parse')

register_args("gdn", "window_size", "model.network.init_args.input_dim", apply_on='parse')
register_args("fedformer", "window_size", "model.network.init_args.window_size", apply_on='parse')
register_args("hpad", "window_size", "model.network.init_args.window_size", apply_on='parse')
register_args("timesnet", "window_size", "model.network.init_args.window_size", apply_on='parse')
register_args("neutralad", "window_size", "model.network.init_args.seq_len", apply_on='parse')
register_args("physdiff", "window_size", "model.network.init_args.win_size", apply_on="parse")
register_args("lstm_p", "prediction_horizon", "model.network.init_args.prediction_horizon", apply_on='parse')
register_args("energyad_autoformer", "window_size", "model.network.init_args.window_size", apply_on='parse')
register_args("energyad_fedformer", "window_size", "model.network.init_args.window_size", apply_on='parse')

register_args("lnt", "data.num_features", "model.network.init_args.ts_channels")
register_args("lnt", "model.losses", "model.detector.init_args.loss", apply_on='parse')
register_args("lnt", "window_size", "model.network.init_args.seq_len", apply_on='parse')

register_args("kmeans", "data.batch_size", "model.detector.init_args.batch_size", apply_on='parse')

register_args("kan_ad", "data.num_features", "model.network.init_args.input_dim")
register_args("kan_ad", "window_size", "model.network.init_args.window_size", apply_on='parse')
register_args("kan_ad", "prediction_horizon", "model.network.init_args.prediction_horizon", apply_on='parse')

register_args("igad", "data.num_features", "model.network.init_args.input_dim")
register_args("igad", "window_size", "model.network.init_args.window_size", apply_on='parse')

register_args("oraclead", "data.num_features", "model.network.init_args.input_dim")
register_args("oraclead", "model.losses", "model.detector.init_args.loss", apply_on='parse')
register_args("oraclead", "window_size", "model.network.init_args.win_size", apply_on="parse")

register_args("anomaly_transformer", "data.num_features", "model.network.init_args.enc_in")
register_args("anomaly_transformer", "data.num_features", "model.network.init_args.c_out")
register_args("anomaly_transformer", "window_size", "model.network.init_args.win_size", apply_on="parse")

register_args("alora", "data.num_features", "model.network.init_args.input_dim")
register_args("alora", "window_size", "model.network.init_args.win_size", apply_on="parse")

register_args("dcdetector", "data.num_features", "model.network.init_args.input_dim")
register_args("dcdetector", "window_size", "model.network.init_args.win_size", apply_on="parse")

register_args("catch", "data.num_features", "model.network.init_args.input_dim")
register_args("catch", "window_size", "model.network.init_args.seq_len", apply_on="parse")
register_args("catch", "window_size", "model.detector.init_args.seq_len", apply_on="parse")

register_args("dada", "window_size", "model.detector.init_args.seq_len", apply_on="parse")

register_args("rtdetector", "data.num_features", "model.network.init_args.feats")
register_args("rtdetector", "window_size", "model.network.init_args.window_size", apply_on="parse")

register_args("carots", "data.num_features", "model.network.init_args.input_dim")
register_args("carots", "model.losses", "model.detector.init_args.loss", apply_on="parse")
register_args("carots", "window_size", "model.network.init_args.win_size", apply_on="parse")
register_args("carots", "window_size", "model.detector.init_args.win_size", apply_on="parse")

register_args("paano", "window_size", "model.network.init_args.patch_size", apply_on="parse")
register_args("paano", "window_size", "model.detector.init_args.patch_size", apply_on="parse")

register_args("scatterad", "data.num_features", "model.network.init_args.input_dim")
register_args("scatterad", "window_size", "model.network.init_args.win_size", apply_on="parse")

for _neural_model in (
    "itransformer",
    "modern_tcn",
    "dualtf",
    "patchtst",
    "dlinear",
    "nlinear",
    "tfad",
    "autoencoder",
):
    register_args(_neural_model, "data.num_features", "model.network.init_args.input_dim")
    register_args(_neural_model, "window_size", "model.network.init_args.seq_len", apply_on="parse")
    register_args(_neural_model, "window_size", "model.detector.init_args.seq_len", apply_on="parse")
