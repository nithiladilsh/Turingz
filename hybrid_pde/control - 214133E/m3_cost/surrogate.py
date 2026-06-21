import os
import numpy as np


class Surrogate:
    name = "surrogate"

    def predict_rollout(self, ic):
        raise NotImplementedError

    def available_ics(self):
        return None

    def step_cost_fn(self):
        return None


class CachedSurrogate(Surrogate):
    def __init__(self, npz_path, name="DeepONet(cached)"):
        d = np.load(npz_path, allow_pickle=True)
        self._fields = {int(d["ic_index"]): d["u_pred"].astype(np.float64)}
        self.x = d["x"]; self.t = d["t"]
        self.name = name

    def predict_rollout(self, ic_index):
        if ic_index not in self._fields:
            raise KeyError(
                f"CachedSurrogate has no field for IC {ic_index}; "
                f"available={list(self._fields)}. Use TorchSurrogate for others.")
        return self._fields[ic_index]

    def available_ics(self):
        return sorted(self._fields)


class TorchSurrogate(Surrogate):
    def __init__(self, kind, root, device=None):
        self.kind = kind
        self.root = root
        self.name = {"fno": "FNO", "deeponet": "DeepONet", "pinn": "PINN"}[kind]
        self._model = None
        self._device = device
        self._load()

    def _load(self):
        import torch
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = getattr(self, f"_load_{self.kind}")()

    def _load_fno(self):
        import torch
        from neuralop.models import FNO
        cfg = torch.load(os.path.join(self.root, "results", "fno", "fno_config.pt"),
                         weights_only=False)
        model = FNO(n_modes=(cfg["n_modes"],), hidden_channels=cfg["hidden_channels"],
                    in_channels=cfg.get("in_channels", 3), out_channels=1)
        sd = torch.load(os.path.join(self.root, "results", "fno", "fno.pt"),
                        map_location=self._device, weights_only=False)
        model.load_state_dict(sd)
        self._cfg = cfg
        return model.to(self._device).eval()

    def _load_deeponet(self):
        raise NotImplementedError(
            "DeepONet host loader: restore results/deeponet/model.pt into "
            "DeepONetDDE.net; use CachedSurrogate meanwhile.")

    def _load_pinn(self):
        raise NotImplementedError(
            "PINN is per-IC (results/pinn/pinn_ic*.pt); only train ICs 0..9 exist.")

    def _fno_inputs(self, ic):
        import torch
        from . import groundtruth as G
        nt, nx = G.T_GRID.shape[0], G.X.shape[0]
        ch_ic = np.broadcast_to(ic, (nt, nx))
        ch_t = np.broadcast_to(G.T_GRID[:, None], (nt, nx))
        ch_x = np.broadcast_to(G.X[None, :], (nt, nx))
        inp = np.stack([ch_ic, ch_t, ch_x], axis=1).astype("float32")
        return torch.from_numpy(inp).to(self._device)

    def predict_rollout(self, ic_index):
        from . import groundtruth as G
        ic = G.make_ics()[ic_index]
        if self.kind == "fno":
            import torch
            with torch.no_grad():
                out = self._model(self._fno_inputs(ic))
            return out.squeeze(1).cpu().numpy().astype("float64")
        raise NotImplementedError(f"predict_rollout for kind={self.kind}")

    def step_cost_fn(self):
        from . import groundtruth as G
        ic = G.make_ics()[900]
        if self.kind == "fno":
            import torch
            x1 = self._fno_inputs(ic)[:1]

            def one_step():
                with torch.no_grad():
                    return self._model(x1)
            return one_step
        return None


def get_surrogate(kind, root, prefer_torch=True):
    from . import config as C
    if prefer_torch:
        try:
            return TorchSurrogate(kind, root)
        except Exception as e:
            if kind != "deeponet":
                raise
            print(f"[surrogate] torch unavailable ({type(e).__name__}); "
                  f"using cached DeepONet field.")
            return CachedSurrogate(C.DEEPONET_FIELD)
    return CachedSurrogate(C.DEEPONET_FIELD)
