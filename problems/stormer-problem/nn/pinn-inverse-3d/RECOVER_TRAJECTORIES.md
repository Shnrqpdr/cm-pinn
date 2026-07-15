# Recuperação de trajetórias a partir dos modelos salvos

Os experimentos salvam o modelo treinado (`.pth`) com o `history` (loss, alpha1 por época) e a `config`, mas **não** salvam os arrays da trajetória reconstruída. Porém é possível reconstruí-los carregando o modelo e chamando `predict()`.

## Experimentos disponíveis

Todos em `results/` relativo a este diretório:

### Equatorial (Próton 1)
| Diretório | Descrição | Versão final? |
|-----------|-----------|---------------|
| `equatorial/proton1/expA/` | Exp A — tentativa inicial | Não |
| `equatorial/proton1/expA_v3_short/` | Exp A — v3 | Não |
| `equatorial/proton1/expA_v4_2ndorder/` | Exp A — v4 2nd order | Não |
| `equatorial/proton1/expA_v6_final/` | Exp A — v6 sem normalização | Não |
| `equatorial/proton1/expA_v7_normalized/` | **Exp A — versão final** | **Sim** |
| `equatorial/proton1/expB_30pct/` | **Exp B — 30% sparse** | **Sim** |
| `equatorial/proton1/expC1_noise0.01/` | **Exp C1 — noise 0.01** | **Sim** |
| `equatorial/proton1/expC2_noise0.05/` | **Exp C2 — noise 0.05** | **Sim** |
| `equatorial/proton1/expC3_noise0.1/` | **Exp C3 — noise 0.1** | **Sim** |

### 3D (Próton 1)
| Diretório | Descrição | Versão final? |
|-----------|-----------|---------------|
| `3d/proton1/expA_expA/` | Exp A — primeira tentativa (full ODE, alpha1 biased) | Não |
| `3d/proton1/expA_expA_v2/` | **Exp A — versão final (phi-only Phase 1)** | **Sim** |

## O que está salvo no `.pth`

```python
{
    "model_state_dict": ...,     # pesos da rede
    "alpha1_estimated": float,   # alpha1 final
    "alpha1_true": float,        # alpha1 verdadeiro
    "alpha1_rel_error": float,   # erro relativo final
    "history": {                 # dados por época (para gráficos de loss/convergência)
        "epoch": [...],
        "total": [...],
        "data": [...],
        "ode": [...],
        "alpha1": [...],
    },
    "config": {                  # todos os hiperparâmetros
        "mode": "equatorial" | "3d",
        "alpha1_init": float,
        "n_hidden": int, "n_neurons": int,
        "n_frequencies": int,
        "warmup_epochs": int, "alpha1_epochs": int,
        "adam_epochs": int, "adam_lr": float,
        "alpha1_lr": float,
        "w_data": float, "w_ode": float,
        "obs_fraction": float, "noise_std": float,
    },
}
```

## Como recuperar a trajetória reconstruída

### Script de recuperação

```python
import torch
import numpy as np
from pinn_stormer_3d_inverse import Stormer3DInversePINN, Stormer3DInverseTrainer

def recover_trajectory(pth_path, dataset_path):
    """Carrega modelo salvo e reconstrói a trajetória.

    Parâmetros
    ----------
    pth_path : str
        Caminho para o .pth (ex: "results/equatorial/proton1/expA_v7_normalized/pinn_inverse_model.pth")
    dataset_path : str
        Caminho para o .npz correspondente:
        - Equatorial: "data/dataset_equatorial_proton1.npz"
        - 3D: "data/dataset_3d_proton1.npz"

    Retorna
    -------
    dict com: t_ref, rho_pred, phi_pred, rho_ref, phi_ref, history, config
              (e Z_pred, Z_ref para modo 3d)
    """
    torch.set_default_dtype(torch.float64)
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # Recriar modelo com a mesma arquitetura
    model = Stormer3DInversePINN(
        mode=cfg["mode"],
        n_hidden=cfg["n_hidden"],
        n_neurons=cfg["n_neurons"],
        n_frequencies=cfg["n_frequencies"],
        alpha1_init=cfg["alpha1_init"],  # valor inicial (será sobrescrito)
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Recriar trainer para acessar dados de referência
    trainer = Stormer3DInverseTrainer(
        model, dataset_path, device="cpu",
        w_data=cfg["w_data"], w_ode=cfg["w_ode"],
        obs_fraction=cfg["obs_fraction"], noise_std=cfg["noise_std"],
    )

    # Reconstruir trajetória
    pred = trainer.predict(trainer.t_ref)

    result = {
        "t_ref": trainer.t_ref,
        "rho_pred": pred["rho"],
        "rho_ref": trainer.rho_ref,
        "phi_pred": pred["phi"],
        "phi_ref": trainer.phi_ref,
        "history": ckpt["history"],
        "config": cfg,
        "alpha1_estimated": ckpt["alpha1_estimated"],
        "alpha1_true": ckpt["alpha1_true"],
        "alpha1_rel_error": ckpt["alpha1_rel_error"],
    }

    if cfg["mode"] == "3d":
        result["Z_pred"] = pred["Z"]
        result["Z_ref"] = trainer.Z_ref

    return result


def save_trajectory_npz(result, output_path):
    """Salva trajetória reconstruída em .npz para uso posterior."""
    arrays = {
        "t_ref": result["t_ref"],
        "rho_pred": result["rho_pred"],
        "rho_ref": result["rho_ref"],
        "phi_pred": result["phi_pred"],
        "phi_ref": result["phi_ref"],
        "alpha1_estimated": np.array(result["alpha1_estimated"]),
        "alpha1_true": np.array(result["alpha1_true"]),
        "alpha1_rel_error": np.array(result["alpha1_rel_error"]),
        # history arrays
        "hist_epoch": np.array(result["history"]["epoch"]),
        "hist_total": np.array(result["history"]["total"]),
        "hist_data": np.array(result["history"]["data"]),
        "hist_ode": np.array(result["history"]["ode"]),
        "hist_alpha1": np.array(result["history"]["alpha1"]),
    }
    if "Z_pred" in result:
        arrays["Z_pred"] = result["Z_pred"]
        arrays["Z_ref"] = result["Z_ref"]

    np.savez_compressed(output_path, **arrays)
    print(f"Saved: {output_path}")
```

### Uso: recuperar um experimento

```python
# Equatorial Exp A (versão final)
result = recover_trajectory(
    "results/equatorial/proton1/expA_v7_normalized/pinn_inverse_model.pth",
    "data/dataset_equatorial_proton1.npz",
)

# Salvar como .npz
save_trajectory_npz(result, "results/equatorial/proton1/expA_v7_normalized/trajectory_data.npz")
```

### Uso: recuperar TODOS os experimentos finais

```python
experiments = [
    # (pth_path, dataset_path)
    ("results/equatorial/proton1/expA_v7_normalized/pinn_inverse_model.pth",
     "data/dataset_equatorial_proton1.npz"),
    ("results/equatorial/proton1/expB_30pct/pinn_inverse_model.pth",
     "data/dataset_equatorial_proton1.npz"),
    ("results/equatorial/proton1/expC1_noise0.01/pinn_inverse_model.pth",
     "data/dataset_equatorial_proton1.npz"),
    ("results/equatorial/proton1/expC2_noise0.05/pinn_inverse_model.pth",
     "data/dataset_equatorial_proton1.npz"),
    ("results/equatorial/proton1/expC3_noise0.1/pinn_inverse_model.pth",
     "data/dataset_equatorial_proton1.npz"),
    ("results/3d/proton1/expA_expA_v2/pinn_inverse_model.pth",
     "data/dataset_3d_proton1.npz"),
]

for pth, ds in experiments:
    result = recover_trajectory(pth, ds)
    out = pth.replace("pinn_inverse_model.pth", "trajectory_data.npz")
    save_trajectory_npz(result, out)
```

## Datasets correspondentes

- **Equatorial** (todos os exp equatoriais): `data/dataset_equatorial_proton1.npz`
- **3D** (todos os exp 3D): `data/dataset_3d_proton1.npz`

## Nota sobre experimentos intermediários (v3, v4, v6)

Os checkpoints intermediários (`expA`, `expA_v3_short`, `expA_v4_2ndorder`, `expA_v6_final`) podem ter `config` incompleta ou com campos ligeiramente diferentes. A `config` foi padronizada a partir de `expA_v7_normalized`. Se algum campo estiver ausente, usar os defaults:
- `mode`: `"equatorial"` (todos os intermediários são equatoriais)
- `n_hidden`: 4, `n_neurons`: 128
- `n_frequencies`: 50 (v6, v7) ou 10 (versões anteriores)
- `obs_fraction`: 1.0, `noise_std`: 0.0
