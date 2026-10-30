"""Training and evaluation loop. Owner: Ashritha (Trello: [S1] Training + evaluation loop).

Notebooks call run_experiment(); keep its name and arguments the same.
"""


def make_optimizer(name: str, params, lr: float):
    """name is "sgd" (plain, no momentum) or "radam".

    TODO (Ashritha): torch.optim.SGD(params, lr=lr) / torch.optim.RAdam(params, lr=lr)
    """
    raise NotImplementedError


def train_one_epoch(model, loader, optimizer, loss_fn):
    """Return average training loss for one epoch. TODO (Ashritha)."""
    raise NotImplementedError


def evaluate(model, loader, loss_fn):
    """Return dict with loss, acc and auc (one-vs-rest, sklearn roc_auc_score). TODO (Ashritha)."""
    raise NotImplementedError


def run_experiment(optimizer_name: str, lr: float, epochs: int,
                   batch_size: int = 64, seed: int = 42,
                   out_csv: str = None):
    """Train from init.pt and log one row per epoch to results/<out_csv>.

    CSV columns: epoch, train_loss, val_loss, val_acc, val_auc,
                 epoch_seconds, optimizer_state_bytes

    TODO (Ashritha):
      - set_seed(seed); loaders = get_loaders(batch_size, seed)
      - model = load_init_weights(build_model())
      - time each epoch with time.perf_counter()
      - optimizer_state_bytes(optimizer) after the first epoch
      - save the CSV with pandas
    """
    raise NotImplementedError
