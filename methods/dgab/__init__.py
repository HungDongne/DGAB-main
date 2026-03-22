class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0, mode="max"):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.best_value = None
        self.is_earlystop = False
        self.count = 0
        self.best_epoch = None
        self.best_model = None
        self.best_state_dict = None

    def earlystop(self, score, model, epoch=None):
        if self.mode == "max":
            value = score
        else:
            value = -score

        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.best_model = model
            if self.verbose:
                print(
                    f"EarlyStopper: initial best score = {score:.6f} (epoch {epoch})"
                )
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print(
                    f"EarlyStopper count: {self.count:02d} "
                    f"(current={score:.6f}, best={self._raw_best():.6f} at epoch {self.best_epoch})"
                )
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_epoch = epoch
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.best_model = model
            self.count = 0
            if self.verbose:
                print(
                    f"EarlyStopper: new best score = {score:.6f} (epoch {epoch})"
                )

    def _raw_best(self):
        if self.best_value is None:
            return None
        return self.best_value if self.mode == "max" else -self.best_value

    def get_best_model(self, device="cpu"):
        if self.best_state_dict is not None:
            self.best_model.load_state_dict(self.best_state_dict)
        return self.best_model.to(device)
