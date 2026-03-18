# PINN Inverso — Experimentos

## Todos os experimentos concluídos

### Case 1 (theta0=pi/3, p_theta0=0, p_phi0=0.394) — k_init=1.0
- [x] Exp A (100%, sem ruído)
- [x] Exp B 30%
- [x] Exp B 15%
- [x] Exp B 5%
- [x] Exp C1 (noise=0.01)
- [x] Exp C2 (noise=0.05)
- [x] Exp C3 (noise=0.1)

### Case 2 (theta0=pi/4, p_theta0=0, p_phi0=0.394, regime two_hemispheres) — k_init=1.0
- [x] Exp A (100%, sem ruído)
- [x] Exp B 15%
- [x] Exp B 5%
- [x] Exp C1 (noise=0.01)
- [x] Exp C2 (noise=0.05)
- [x] Exp C3 (noise=0.1)

### Case 3 (theta0=0.6, p_theta0=0.1, p_phi0=0.25, regime one_hemisphere) — k_init=0.3
- [x] Exp A (100%, sem ruído)
- [x] Exp B 15%
- [x] Exp B 5%
- [x] Exp C1 (noise=0.01)
- [x] Exp C2 (noise=0.05)
- [x] Exp C3 (noise=0.1)

## Notas importantes

- **Case 3 requer k_init=0.3**: Com k_init=1.0 (default), o otimizador fica preso num
  mínimo local em k≈0.667. Abordando por baixo (k_init=0.3), converge sem problemas.
  Causa: paisagem de loss com mínimo local quando p_theta0 != 0.
- Resultados documentados em `notes/stormer-pinn-inverse.md`.
