# PINN Inversa: Problema de Störmer 3D (sem vínculo)

**Data**: 2026-03-19
**Referência**: `references/stormer-problem/relatorio_final_pibic_pedro.pdf`
**Pré-requisito**: PINN inversa na esfera (Issue #5) — validada com sucesso

---

## 1. Objetivo

Identificar o parâmetro de acoplamento magnético $\alpha_1 = \frac{qM_z}{mr_0^3}$ a partir de dados de trajetória gerados pelo solver Störmer-Verlet em C. Duas fases:

- **Fase 1**: Caso equatorial ($Z = 0$) — 1 ODE radial + equação algébrica para $\varphi$
- **Fase 2**: Caso 3D completo — 2 ODEs acopladas ($\rho, Z$) + equação para $\varphi$

O valor verdadeiro é $\alpha_1 = 3.037 \times 10^3 \text{ s}^{-1}$ (prótons na magnetosfera terrestre).

---

## 2. Diferenças em relação ao caso na esfera

| Aspecto | Esfera (Issue #5) | 3D livre (este) |
|---------|-------------------|-----------------|
| Graus de liberdade | 2 ($\theta, \varphi$) | 2 ($\rho, \varphi$) equatorial / 3 ($\rho, Z, \varphi$) 3D |
| Solução analítica | Sim (Piña & Cortés) | Não — dados numéricos |
| Fonte dos dados | `generate_dataset_inverse.py` (analítico) | Solver C `sv_equatorial.c` / `sv_3d.c` |
| Formulação ODE | $z = \cos\theta$, singularity-free | Cilíndricas $(\rho, \varphi, Z)$ |
| Parâmetro a identificar | $k = \mu_0 q m / (4\pi R)$ | $\alpha_1 = qM_z / (mr_0^3)$ |
| Constante conservada | $p_\varphi$, energia $(A, B)$ | $c_2$ (momento angular canônico), energia |
| Escalas físicas | Adimensional (paper) | Semi-dimensional ($M \sim 10^{-27}$, $\alpha_1 \sim 10^3$) |

---

## 3. Física do problema

### 3.1 Parâmetros físicos

- $M = 1.6726219 \times 10^{-27}$ kg (massa do próton)
- $\alpha_1 = 3.037 \times 10^3$ s$^{-1}$ (acoplamento magnético, escalado por $r_0$)
- $r_0 = 6{,}378{,}136$ m (raio terrestre — usado para adimensionalizar coordenadas)

As coordenadas $(\rho, Z)$ nos solvers já estão em unidades de $r_0$ (adimensionais).
As velocidades $(\dot{\rho}, \dot{Z})$ estão em $r_0$/s.

### 3.2 Momento angular canônico

$$c_2 = M\rho^2\dot{\varphi} + \frac{M\alpha_1\rho^2}{R^3}$$

onde $R = \sqrt{\rho^2 + Z^2}$. Para o caso equatorial ($Z = 0$, $R = \rho$):

$$c_{20} = M\rho_0^2\dot{\varphi}_0 + \frac{M\alpha_1}{\rho_0}$$

**Ponto-chave**: $c_{20}$ (e $c_2$) dependem de $\alpha_1$. Durante o treino, como $\alpha_1$ é parâmetro treinável, $c_{20}(\alpha_1)$ deve ser recalculado a cada step, mantendo o grafo computacional.

### 3.3 Caso equatorial — EDOs

$$\ddot{\rho} = \frac{c_{20}^2}{M^2\rho^3} - \frac{3\alpha_1 c_{20}}{M\rho^4} + \frac{2\alpha_1^2}{\rho^5}$$

$$\dot{\varphi} = \frac{c_{20}}{M\rho^2} - \frac{\alpha_1}{\rho^3}$$

Energia conservada:

$$E = \frac{1}{2}\dot{\rho}^2 + \frac{c_{20}^2}{2M^2\rho^2} - \frac{\alpha_1 c_{20}}{M\rho^3} + \frac{\alpha_1^2}{2\rho^4}$$

### 3.4 Caso 3D — EDOs

$$\ddot{\rho} = \frac{c_2^2}{M^2\rho^3} + \frac{3\alpha_1^2\rho^3}{R^8} - \frac{\alpha_1^2\rho}{R^6} - \frac{3\alpha_1 c_2\rho}{MR^5}$$

$$\ddot{Z} = \frac{3\alpha_1^2\rho^2 Z}{R^8} - \frac{3\alpha_1 c_2 Z}{MR^5}$$

$$\dot{\varphi} = \frac{c_2}{M\rho^2} - \frac{\alpha_1}{R^3}$$

Energia conservada:

$$H_\text{eff} = \frac{1}{2}(\dot{\rho}^2 + \dot{Z}^2) + \frac{c_2^2}{2M^2\rho^2} - \frac{\alpha_1 c_2}{MR^3} + \frac{\alpha_1^2\rho^2}{2R^6}$$

---

## 4. Formulação e normalização — Evolução

### 4.1 Tentativa 1: First-order com $\tau$-scaling (FALHOU)

**Ideia original** (adaptada da esfera): rede prediz $(\rho, w_\rho, \varphi)$ onde $w_\rho = d\rho/d\tau$ é uma velocidade adimensional. Escalas $\tau_\text{scale}$, $S = T_\text{obs}/\tau_\text{scale}$ dependem de $\alpha_1$.

$$c_{20} = M\rho_0^2\dot{\varphi}_0 + \frac{M\alpha_1}{\rho_0}, \quad E = \frac{1}{2}\dot{\rho}_0^2 + V_\text{eff}(\rho_0; \alpha_1)$$

$$\tau_\text{scale} = \frac{\rho_0}{\sqrt{2E}}, \quad S = \frac{T_\text{obs}}{\tau_\text{scale}}$$

Com valores do Próton 1 e $\alpha_1 = 3037$: $c_{20}/M = 1102.3$, $E = 500$, $\tau_\text{scale} = 0.095$ s, $S = 21.1$.

**Residuais:**

$$r_1 = \frac{d\rho}{dt_\text{norm}} - S \cdot w_\rho, \quad r_2 = \frac{dw_\rho}{dt_\text{norm}} - S \cdot f(\rho), \quad r_3 = \frac{d\varphi}{dt_\text{norm}} - S \cdot \tau_\text{scale} \cdot g(\rho)$$

onde $f$ e $g$ são as forças normalizadas por $\tau_\text{scale}^2$.

**Por que falhou:** $w_\rho$ é uma variável auxiliar **não observável nos dados**. Durante o warmup (data-only), a rede aprende $\rho(t)$ e $\varphi(t)$ bem (data loss $\sim 10^{-6}$), mas $w_\rho$ sai com valores aleatórios. Quando a loss ODE ativa:

- $r_1$ é enorme porque $w_\rho$ é aleatório → ODE loss inicial $\sim 10^6$
- O gradiente ODE destrói o data fit: data loss salta de $10^{-6}$ para $\sim 140$
- $\alpha_1$ não converge — passa pelo valor correto e continua derivando

**Tentativas de correção que não funcionaram:**
- Kinematic consistency loss no warmup ($r_1$ com $\alpha_1$ congelado): degradou data loss de $10^{-6}$ para $0.024$ por competição de objetivos
- Warmup em duas fases (data-only → data+kinematic): mesma degradação do data fit
- ODE weight ramp-up (0.01 → 1.0): $\alpha_1$ convergiu parcialmente para $\sim 3380$ (erro 11%) mas desestabilizou quando o peso chegou a 1.0

### 4.2 Tentativa 2: First-order + Joint Adam (FALHOU)

Mesmo com $w_\rho$ constrainido, a **otimização conjunta** rede + $\alpha_1$ via Adam é o problema fundamental. A rede compensa $\alpha_1$ errado ajustando seus outputs. O otimizador encontra mínimos espúrios onde $\alpha_1$ é errado mas a combinação data+ODE é localmente mínima.

**Evidências:**
- $\alpha_{1,\text{init}} = 5000$: convergiu para $\alpha_1 \approx 452$ (erro 85%), $\alpha_1 \approx 1286$ (erro 58%), ou $\alpha_1 \approx 298$ (erro 90%) dependendo dos pesos
- $\alpha_{1,\text{init}} = 1500$: divergiu para $\alpha_1 \approx 935$ (erro 69%) — direção errada
- Mesmo com `w_data=100`: $\alpha_1$ deslizou de 3038 → 2818 (erro 7.2%) durante joint Adam

**Diagnóstico de paisagem:** scan de $\alpha_1$ sobre os dados verdadeiros (trajetória do solver, não da rede) mostrou mínimo claro do ODE residual em $\alpha_1 \approx 3102$ (perto do verdadeiro 3037). O problema não são mínimos locais no espaço de $\alpha_1$, mas mínimos espúrios no espaço conjunto (pesos da rede, $\alpha_1$).

### 4.3 Formulação final: Second-order M-free (FUNCIONA)

**Três mudanças cruciais:**

#### 4.3.1 Eliminação de $M$ das computações

$M$ cancela analiticamente em todas as equações de força e de $\dot{\varphi}$. Definindo $\hat{c}_{20} = c_{20}/M$:

$$\hat{c}_{20} = \rho_0^2\dot{\varphi}_0 + \frac{\alpha_1}{\rho_0}$$

As equações ficam M-free:

$$\ddot{\rho} = \frac{\hat{c}_{20}^2}{\rho^3} - \frac{3\alpha_1\hat{c}_{20}}{\rho^4} + \frac{2\alpha_1^2}{\rho^5}$$

$$\dot{\varphi} = \frac{\hat{c}_{20}}{\rho^2} - \frac{\alpha_1}{\rho^3}$$

Com $\rho_0 = 3$, $\dot{\varphi}_0 = 10$, $\alpha_1 = 3037$: $\hat{c}_{20} = 90 + 1012.3 = 1102.3$.

**Verificação numérica:** força em $\rho = 3$:
- $\hat{c}_{20}^2/\rho^3 = 1102.3^2/27 \approx 45{,}000$
- $3\alpha_1\hat{c}_{20}/\rho^4 = 3 \times 3037 \times 1102.3/81 \approx 124{,}000$
- $2\alpha_1^2/\rho^5 = 2 \times 3037^2/243 \approx 75{,}800$
- Força líquida $\approx -3{,}200$ s$^{-2}$

#### 4.3.2 Formulação de segunda ordem (sem $w_\rho$)

A rede outputa apenas $(\rho, \varphi)$ — as grandezas diretamente observáveis. As derivadas são computadas via autograd:

$$\frac{d\rho}{dt_\text{norm}}, \quad \frac{d^2\rho}{dt_\text{norm}^2}, \quad \frac{d\varphi}{dt_\text{norm}}$$

Residuais em tempo normalizado ($t_\text{norm} = t/T_\text{obs}$):

$$r_\rho = \frac{d^2\rho}{dt_\text{norm}^2} - T_\text{obs}^2 \cdot \text{force}(\rho, \alpha_1)$$

$$r_\varphi = \frac{d\varphi}{dt_\text{norm}} - T_\text{obs} \cdot \text{phi\_rate}(\rho, \alpha_1)$$

Para $T_\text{obs} = 2$ s: $r_\rho \sim T_\text{obs}^2 \times 3200 = 12{,}800$, $r_\varphi \sim T_\text{obs} \times 10 = 20$.

#### 4.3.3 Normalização dos residuais para $O(1)$

Os residuais brutos têm escalas muito diferentes ($r_\rho \sim 10^4$, $r_\varphi \sim 10^1$). Dividir por escalas de referência calculadas nas condições iniciais:

$$\sigma_\rho = T_\text{obs}^2 \times |\text{force}(\rho_0, \alpha_{1,\text{init}})| \approx 21{,}022$$

$$\sigma_\varphi = T_\text{obs} \times |\text{phi\_rate}(\rho_0, \alpha_{1,\text{init}})| \approx 20$$

Residuais normalizados:

$$\hat{r}_\rho = r_\rho / \sigma_\rho, \quad \hat{r}_\varphi = r_\varphi / \sigma_\varphi$$

Agora $\hat{r}_\rho^2 + \hat{r}_\varphi^2 \sim O(1)$ quando o treinamento não convergiu, e $\sim O(10^{-6})$ quando convergiu. Os pesos $\omega_\text{data}$ e $\omega_\text{ode}$ passam a ser diretamente comparáveis.

**Sem normalização:** ODE loss inicial $\sim 10^9$ com $\omega_\text{ode} = 0.1$ → contribuição $\sim 10^8$, data loss $\sim 10^{-6}$. Impossível balancear.

**Com normalização:** ODE loss inicial $\sim 3.5$ com $\omega_\text{ode} = 1.0$ → contribuição $\sim 3.5$, data loss $\sim 10^{-6}$. Balanceamento natural.

### 4.4 Extensão para o caso 3D

Extensão direta da formulação M-free de segunda ordem. A rede prediz 3 saídas: $(\rho, Z, \varphi)$.

$$\hat{c}_2 = \rho_0^2\dot{\varphi}_0 + \frac{\alpha_1\rho_0^2}{R_0^3}$$

3 residuais: $\hat{r}_\rho$ para $\ddot{\rho}$, $\hat{r}_Z$ para $\ddot{Z}$, $\hat{r}_\varphi$ para $\dot{\varphi}$. Todos normalizados por escalas de referência.

---

## 5. Dados

### 5.1 Geração de dados — Caso equatorial

Usar o solver C existente **sem modificações**: `problems/stormer-problem/simulation/no_constraint_case/equatorial_case/sv_equatorial.c`

**Argumentos do solver:**
```
./sv_equatorial T_final rho0 drho0 phi0 dphi0 outputParticle.dat outputPotential.dat
```

O solver calcula $c_{20}$ internamente: `c20 = dphi*M*(rho[0]*rho[0]) + M*alpha1*(1/rho[0])`.

**Formato de saída** (`outputParticle.dat`):
```
n   x   y   z
0   3.0 0.0 0.0
1   ...
```

Onde $(x, y)$ são coordenadas Cartesianas (plano equatorial), $z = 0$ sempre.

**Condições iniciais (Figura 5 do relatório):**

| Próton | $\rho_0$ | $\dot{\rho}_0$ | $\varphi_0$ | $\dot{\varphi}_0$ | Regime |
|--------|----------|-----------------|-------------|-------------------|--------|
| 1 | 3.0 | 10.0 | 0.0 | 10.0 | Órbita contida |
| 2 | 3.0 | 80.0 | $3\pi/2$ | 10.0 | Alta velocidade radial |
| 3 | 3.0 | 100.0 | $\pi$ | 10.0 | Velocidade radial muito alta |

**Começar com o Próton 1** (dinâmica mais contida, melhor para validação).

$\Delta t_\text{solver} = 0.0001$ s. Usar $T_\text{final} = 2$ s → 20{,}000 timesteps.

### 5.2 Pipeline de pré-processamento

Script Python `generate_dataset_equatorial.py`:

1. **Compilar** `sv_equatorial.c` → `sv_equatorial`
2. **Rodar** solver com ICs do Próton 1
3. **Ler** `outputParticle.dat` → arrays $(n, x, y)$
4. **Converter** para cilíndricas:
   - $\rho_i = \sqrt{x_i^2 + y_i^2}$
   - $\varphi_i = \text{atan2}(y_i, x_i)$
   - Aplicar `np.unwrap` em $\varphi$ para remover descontinuidades em $\pm\pi$
   - $t_i = n_i \times \Delta t$
5. **Subsample** para N pontos de observação (ex: 1000 pontos igualmente espaçados) — o solver gera 20k pontos, mas não precisamos de todos para treino
6. **Gerar pontos de colocação** (para ODE loss): N_coll pontos uniformemente distribuídos em $[0, T_\text{obs}]$, independentes dos dados
7. **Salvar** `.npz` com:
   - Metadados: `M`, `alpha1_true`, `rho0`, `drho0`, `phi0`, `dphi0`, `T_final`, `dt_solver`
   - Observações: `t_obs`, `rho_obs`, `phi_obs`
   - Referência (todos os 20k pontos): `t_ref`, `rho_ref`, `phi_ref`
   - Colocação: `t_collocation`

### 5.3 Geração de dados — Caso 3D

Solver: `problems/stormer-problem/simulation/no_constraint_case/3d_case/sv_3d.c`

**Argumentos do solver:**
```
./sv3d T_final rho0 drho0 phi0 dphi0 z0 dz0 outputParticle.dat outputPhaseSpace.dat
```

**Nota**: no solver 3D, `drho` e `dz` são armazenados como **momentos** ($p_\rho = M\dot{\rho}$), mas o argumento de entrada é a **velocidade**. O solver faz `drho[0] = M*atof(argv[3])`.

**Condições iniciais (Figura 7 do relatório):**

| Próton | $\rho_0$ | $\dot{\rho}_0$ | $\varphi_0$ | $\dot{\varphi}_0$ | $Z_0$ | $\dot{Z}_0$ |
|--------|----------|-----------------|-------------|-------------------|--------|-------------|
| 1 | 3.0 | 10.0 | 0.0 | 10.0 | 0.5 | 0.0 |
| 2 | 3.0 | 80.0 | $\pi$ | 10.0 | 0.5 | 0.0 |
| 3 | 3.0 | 100.0 | 5.26 | 10.0 | 0.5 | 0.0 |

$\Delta t_\text{solver} = 0.0002$ s (do código C, `deltaT = 0.0002`). Usar $T_\text{final} = 2$ s → 10{,}000 timesteps.

**Nota sobre $\Delta t$**: o relatório menciona $\Delta t = 0.00001$ para o caso 3D (Figura 7), mas o código C usa `deltaT = 0.0002`. Verificar se o código foi atualizado após o relatório, ou se é preciso alterar. Na dúvida, testar com o $\Delta t$ do código e validar conservação de energia.

---

## 6. Arquitetura da PINN (versão final — 2nd-order M-free)

### 6.1 Diretório

```
problems/stormer-problem/nn/pinn-inverse-3d/
├── generate_dataset_equatorial.py  # Compila solver, gera dados equatoriais
├── generate_dataset_3d.py          # Compila solver, gera dados 3D
├── pinn_stormer_3d_inverse.py      # Modelo + Trainer (equatorial E 3D)
├── train_inverse_equatorial.py     # Script de treino — Fase 1
├── train_inverse_3d.py             # Script de treino — Fase 2
├── data/                           # Datasets .npz
└── results/                        # Modelos, plots, logs
```

### 6.2 Rede neural

- **Fourier features**: $n_\text{frequencies} = 50$ (input: $t_\text{norm}$, output: $1 + 100 = 101$ features). O valor 50 é necessário porque a trajetória do Próton 1 com $T_\text{obs} = 2$ s contém ~44 oscilações em $\rho(t)$ — com $n_\text{frequencies} = 10$, a rede só consegue representar ~10 ciclos.
- **MLP**: 4 camadas ocultas, 128 neurônios, ativação tanh
- **Output**: 2 componentes (equatorial: $\rho, \varphi$) ou 3 (3D: $\rho, Z, \varphi$). **Sem variáveis auxiliares de velocidade** — todas as derivadas são computadas via autograd.
- **float64** ao longo de toda a rede
- **$\alpha_1$ treinável** via exp: $\alpha_1 = e^{\log\_\alpha_1}$, garantindo $\alpha_1 > 0$. Inicializado como $\log\_\alpha_1 = \ln(\alpha_{1,\text{init}})$.

### 6.3 Modelo (`pinn_stormer_3d_inverse.py`)

**Classe `Stormer3DInversePINN`:**
- `__init__`: recebe `mode="equatorial"` ou `mode="3d"`, `alpha1_init`, config da rede
- `forward(t_norm)`: retorna 2 saídas (equatorial: $\rho, \varphi$) ou 3 (3D: $\rho, Z, \varphi$)
- `get_alpha1()`: retorna $\alpha_1 = \exp(\log\_\alpha_1)$
- Inicialização Xavier para pesos

**Classe `Stormer3DInverseTrainer`:**
- `__init__`: carrega dataset `.npz`, configura device, pesos de loss. Calcula escalas de normalização ODE ($\sigma_\rho$, $\sigma_\varphi$) no $\alpha_{1,\text{init}}$.
- `_c20_hat(alpha1)`: calcula $\hat{c}_{20}(\alpha_1) = \rho_0^2\dot{\varphi}_0 + \alpha_1/\rho_0$ (M-free)
- `loss_data()`: MSE entre $(\rho_\text{NN}, \varphi_\text{NN})$ e observações
- `loss_ode()`: residuais de 2ª ordem normalizados (equatorial: 2 residuais; 3D: 3 residuais). Usa `torch.autograd.grad` duplo para $d^2\rho/dt_\text{norm}^2$.
- `total_loss()`: retorna `(total, l_data, l_ode)` — sem loss de energia
- `predict(t_physical)`: predição para plotagem
- `get_alpha1_error()`: erro relativo em $\alpha_1$

### 6.4 Função `_c20_hat` e loss ODE (M-free, 2ª ordem)

```python
def _c20_hat(self, alpha1):
    """c20_hat = c20/M — proton mass cancels analytically."""
    return self.rho0**2 * self.dphi0 + alpha1 / self.rho0

def loss_ode(self):
    alpha1 = self.model.get_alpha1()
    c_hat = self._c20_hat(alpha1)
    t = self.t_coll_norm  # requires_grad=True

    out = self.model(t)
    rho, phi = out[:, 0:1], out[:, 1:2]

    # Derivadas via autograd
    drho_dt  = autograd.grad(rho, t, ones, create_graph=True)[0]
    dphi_dt  = autograd.grad(phi, t, ones, create_graph=True)[0]
    d2rho_dt2 = autograd.grad(drho_dt, t, ones, create_graph=True)[0]

    # Força M-free
    force = c_hat**2/rho**3 - 3*alpha1*c_hat/rho**4 + 2*alpha1**2/rho**5
    phi_rate = c_hat/rho**2 - alpha1/rho**3

    # Residuais NORMALIZADOS
    r_rho = (d2rho_dt2 - T_obs² * force) / scale_rho  # scale_rho ≈ 21022
    r_phi = (dphi_dt - T_obs * phi_rate) / scale_phi    # scale_phi ≈ 20

    return mean(r_rho² + r_phi²)
```

As escalas de normalização são calculadas uma vez na inicialização:

$$\sigma_\rho = T_\text{obs}^2 \times |\text{force}(\rho_0, \alpha_{1,\text{init}})| \approx 21{,}022$$

$$\sigma_\varphi = T_\text{obs} \times |\dot{\varphi}(\rho_0, \alpha_{1,\text{init}})| \approx 20$$

### 6.5 Positividade de $\alpha_1$

Parametrização via exponencial: $\alpha_1 = e^{\log\_\alpha_1}$.

Para $\alpha_{1,\text{init}} = 5000$: $\log\_\alpha_1 = \ln(5000) \approx 8.517$.

A exponencial garante $\alpha_1 > 0$ e tem gradiente mais estável que softplus para valores grandes. Com Adam `lr=0.01`, a escala logarítmica permite passos proporcionais: $\Delta\alpha_1/\alpha_1 \approx \Delta\log\_\alpha_1$.

---

## 7. Treinamento (versão final — 4 fases desacopladas)

### 7.1 Receita: nunca otimizar rede e $\alpha_1$ conjuntamente

A lição central do debugging (Seção 4.2) é que a otimização conjunta rede + $\alpha_1$ via Adam cria mínimos espúrios. A receita final tem **4 fases**, onde rede e parâmetro **nunca são otimizados ao mesmo tempo**:

1. **Fase 0 — Warmup** (5000 épocas): data loss only, $\alpha_1$ congelado
   - Optimizer: Adam `lr=1e-3` com CosineAnnealing até `eta_min=1e-5`
   - Gradient clipping `max_norm=1.0`
   - A rede aprende $\rho(t)$ e $\varphi(t)$ a partir dos dados do solver
   - Data loss final: $\sim 1.2 \times 10^{-6}$ (equatorial Próton 1)

2. **Fase 1 — $\alpha_1$-only** (5000 épocas): rede congelada, ODE loss only
   - Optimizer: Adam `lr=0.01` com CosineAnnealing até `eta_min=1e-4`
   - Só `model.log_alpha1` tem `requires_grad=True`
   - A loss ODE é calculada nos pontos de colocação com a rede fixa
   - O gradiente flui: `loss_ode → alpha1 → c20_hat → force/phi_rate → residuais`
   - Convergência rápida: $\alpha_1$ vai de 5000 → 3037.24 em ~1000 épocas (erro 0.008%)
   - A rede fixa garante que os residuais refletem a física real, sem compensação

3. **Fase 2 — Refinamento da rede** (10{,}000–15{,}000 épocas): $\alpha_1$ congelado, data + ODE
   - Optimizer: Adam `lr=1e-3` com CosineAnnealing até `eta_min=1e-6`
   - $\alpha_1$ congelado no valor identificado na Fase 1
   - Loss: $\omega_\text{data}\mathcal{L}_\text{data} + \omega_\text{ode}\mathcal{L}_\text{ode}$
   - A rede ajusta seus pesos para satisfazer simultaneamente dados E equações com $\alpha_1$ correto
   - ODE loss cai de $1.7 \times 10^{-2}$ para $5 \times 10^{-6}$ ao longo de 10k épocas
   - Data loss se mantém em $\sim 1.4 \times 10^{-7}$

4. **Fase 3 — L-BFGS** (5 outer steps, `max_iter=20`): refinamento final
   - Todos os parâmetros desbloqueados
   - `lr=1.0`, `history_size=100`, `line_search_fn="strong_wolfe"`
   - `tolerance_grad=1e-12`, `tolerance_change=1e-14`
   - $\alpha_1$ sofre pequeno ajuste: 3037.24 → 3036.94 (erro final 0.002%)

**Por que funciona:** Na Fase 1, a paisagem de $\alpha_1$ (ODE loss vs $\alpha_1$ com rede fixa) tem um mínimo global claro e sem mínimos locais — confirmado por scan de paisagem (ver Seção 4.2). Na otimização conjunta, a rede cria graus de liberdade extras que introduzem mínimos espúrios.

### 7.2 Pesos de loss

$$\mathcal{L} = \omega_\text{data}\mathcal{L}_\text{data} + \omega_\text{ode}\mathcal{L}_\text{ode}$$

Não há loss de energia — a conservação de energia é consequência automática da satisfação das EDOs.

**Com residuais normalizados** (Seção 4.3.3), $\omega_\text{data} = 1.0$ e $\omega_\text{ode} = 1.0$ funcionam diretamente. Os residuais normalizados são $O(1)$ quando não convergidos e $O(10^{-3})$ quando convergidos, comparáveis à data loss.

**Sem normalização** (run v6), $\omega_\text{ode} = 0.1$ resultou em ODE loss $\sim 10^9$ que destruiu o data fit na Fase 2 — $\alpha_1$ convergiu (erro 0.05%) mas a trajetória ficou flat ($\rho$ constante).

### 7.3 Chute inicial de $\alpha_1$

O scan de paisagem confirmou que o mínimo da ODE loss em $\alpha_1$ é único (sem mínimos locais) quando a rede está fixa. O chute $\alpha_{1,\text{init}} = 5000$ ($\sim 1.65\times$ o verdadeiro) funciona bem — convergiu em ~1000 épocas da Fase 1.

A fase $\alpha_1$-only é muito mais robusta ao chute inicial do que a otimização conjunta, que falhava com qualquer chute testado (5000, 1500, 3000 — ver Seção 4.2).

### 7.4 Tratamento de $\varphi$

Usada a **Opção A — $\varphi$ direto**: a rede aprende $\varphi(t)$ crescente sem transformação. Para o Próton 1 com $T_\text{obs} = 2$ s, $\varphi$ varia de 0 a ~20 rad. O Fourier embedding com $n_\text{frequencies} = 50$ é suficiente para representar a parte oscilatória. Opções B e C não foram necessárias.

---

## 8. Experimentos e resultados

### 8.1 Resultado: Equatorial Exp A (Próton 1, dados limpos)

**Configuração:** $\rho_0 = 3.0$, $\dot{\rho}_0 = 10.0$, $\varphi_0 = 0.0$, $\dot{\varphi}_0 = 10.0$, $T_\text{obs} = 2.0$ s, 1000 pontos de observação, 3000 pontos de colocação, $\alpha_{1,\text{init}} = 5000$, $n_\text{frequencies} = 50$, $\omega_\text{data} = 1.0$, $\omega_\text{ode} = 1.0$.

**Run v7_normalized** (versão final com residuais normalizados):

| Fase | Épocas | Data loss final | ODE loss final | $\alpha_1$ | Erro relativo |
|------|--------|-----------------|----------------|------------|---------------|
| 0 (Warmup) | 5000 | $1.25 \times 10^{-6}$ | — | 5000 (frozen) | 64.6% |
| 1 ($\alpha_1$-only) | 5000 | $1.25 \times 10^{-6}$ | $1.75 \times 10^{-2}$ | 3037.24 | 0.008% |
| 2 (Refinamento) | 10000 | $1.40 \times 10^{-7}$ | $5.03 \times 10^{-6}$ | 3037.24 (frozen) | 0.008% |
| 3 (L-BFGS) | 5×20 | $1.95 \times 10^{-7}$ | $3.02 \times 10^{-6}$ | 3036.94 | 0.002% |

**Resultado final:** $\alpha_{1,\text{est}} = 3036.94$, $\alpha_{1,\text{true}} = 3037.00$, erro relativo $= 2.0 \times 10^{-5}$ (0.002%). Tempo total: 1205 s (~20 min).

A trajetória reconstruída sobrepõe-se quase perfeitamente ao solver: erros absolutos $|\Delta\rho| \sim 10^{-4}$, $|\Delta\varphi| \sim 10^{-3}$.

**Comparação com run v6 (sem normalização de residuais):**

| Métrica | v6 ($\omega_\text{ode} = 0.1$, sem normalização) | v7 ($\omega_\text{ode} = 1.0$, com normalização) |
|---------|--------------------------------------------------|--------------------------------------------------|
| $\alpha_1$ erro | 0.076% | 0.002% |
| Data loss final | $6.52 \times 10^{-1}$ | $1.95 \times 10^{-7}$ |
| ODE loss final | $4.66 \times 10^{1}$ | $3.02 \times 10^{-6}$ |
| Trajetória | Flat (rede destruída pelo ODE) | Perfeita |

### 8.2 Experimentos planejados

| Exp | Dados | Ruído | O que testa | Status |
|-----|-------|-------|-------------|--------|
| A | 100% (1000 pts) | $\sigma = 0$ | Convergência básica | **FEITO** (0.002% erro) |
| B 30% | 30% (300 pts) | $\sigma = 0$ | Robustez a esparsidade | Pendente |
| B 5% | 5% (50 pts) | $\sigma = 0$ | Limite de esparsidade | Pendente |
| C1 | 100% | $\sigma = 0.01$ | Ruído baixo | Pendente |
| C2 | 100% | $\sigma = 0.05$ | Ruído moderado | Pendente |
| C3 | 100% | $\sigma = 0.1$ | Ruído alto | Pendente |

Após completar equatorial, estender para 3D (Próton 1 com $Z_0 = 0.5$, $\dot{Z}_0 = 0.0$).

### 8.3 CLI

```bash
# Equatorial — Exp A (configuração validada)
python train_inverse_equatorial.py --n-frequencies 50 --w-ode 1.0 --tag v7_normalized

# Equatorial — variações
python train_inverse_equatorial.py --n-frequencies 50 --w-ode 1.0 --fraction 0.3 --tag expB_30pct
python train_inverse_equatorial.py --n-frequencies 50 --w-ode 1.0 --noise 0.01 --tag expC1
python train_inverse_equatorial.py --n-frequencies 50 --w-ode 1.0 --alpha1-init 1500 --tag init1500

# Parâmetros CLI disponíveis:
#   --proton N            Próton (1, 2, 3)
#   --fraction F          Fração de observações (0.0-1.0)
#   --noise S             Desvio padrão do ruído
#   --alpha1-init V       Chute inicial de alpha1
#   --n-frequencies N     Fourier features (default: 15, usar 50 para T=2s)
#   --n-neurons N         Neurônios por camada (default: 128)
#   --n-hidden N          Camadas ocultas (default: 4)
#   --warmup-epochs N     Épocas de warmup (default: 5000)
#   --alpha1-epochs N     Épocas alpha1-only (default: 5000)
#   --adam-epochs N       Épocas de refinamento (default: 15000)
#   --w-data W            Peso data loss (default: 1.0)
#   --w-ode W             Peso ODE loss (default: 0.1)
#   --lbfgs-epochs N      Steps L-BFGS (default: 5)
#   --tag TAG             Tag para nome do diretório de saída
```

---

## 9. Métricas e plots

### 9.1 Métricas

- $\alpha_{1,\text{est}}$, erro relativo $|\alpha_{1,\text{est}} - \alpha_{1,\text{true}}| / \alpha_{1,\text{true}}$
- Loss total, data, ODE (log-scale) — sem loss de energia (removida na formulação final)
- Valores registrados a cada 500 épocas no `history` dict (salvo no `.pth`)

### 9.2 Plots (gerados automaticamente)

1. **`loss_history.png`**: total, data, ODE vs época (escala log)
2. **`alpha1_convergence.png`**: valor de $\alpha_1$ e erro relativo vs época (2 painéis)
3. **`pinn_vs_solver.png`**: $\rho(t)$ e $\varphi(t)$ — PINN vs solver + erros absolutos (3 painéis)
4. **`trajectory_2d.png`**: trajetória $(x, y)$ no plano equatorial — PINN vs solver

---

## 10. Riscos e mitigações

| Risco | Status | Resolução |
|-------|--------|-----------|
| Escala dos residuais desbalanceada | **RESOLVIDO** | Normalizar $r_i$ por escalas de referência $\sigma_\rho$, $\sigma_\varphi$ (Seção 4.3.3). Sem normalização, a rede é destruída na Fase 2. |
| $\varphi$ crescente dificulta treinamento | **NÃO OCORREU** | $\varphi$ varia de 0 a ~20 rad para $T = 2$ s. Opção A (direto) funciona. |
| Mínimo local em $\alpha_1$ | **RESOLVIDO** | A Fase 1 ($\alpha_1$-only com rede fixa) elimina mínimos espúrios. Scan de paisagem confirmou mínimo único. |
| Otimização conjunta cria mínimos espúrios | **RESOLVIDO** | Nunca otimizar rede + $\alpha_1$ ao mesmo tempo (Seção 4.2). Receita de 4 fases desacopladas. |
| Variável auxiliar $w_\rho$ não observável | **RESOLVIDO** | Eliminada via formulação 2ª ordem (Seção 4.1 → 4.3). |
| $M \sim 10^{-27}$ cria gradientes mal condicionados | **RESOLVIDO** | Formulação M-free: $\hat{c}_{20} = c_{20}/M$ (Seção 4.3.1). |
| Rede não aprende alta frequência de $\rho$ | **RESOLVIDO** | $n_\text{frequencies} = 50$ para cobrir 44 oscilações em $T = 2$ s. |
| $\Delta t$ do solver insuficiente | Baixa | Solver simplético, conservação de energia verificada nos dados. |

---

## 11. Sequência de implementação

### Passo 1: Geração de dados equatorial — **FEITO**
- Script: `generate_dataset_equatorial.py`
- Compila `sv_equatorial.c`, roda Próton 1, converte $(x, y) \to (\rho, \varphi)$, salva `.npz`
- Dataset: `data/dataset_equatorial_proton1.npz` (1000 obs + 3000 coll + 20k ref)

### Passo 2: Modelo PINN equatorial — **FEITO**
- `pinn_stormer_3d_inverse.py` com formulação 2ª ordem M-free
- 2 outputs ($\rho, \varphi$), derivadas via autograd, residuais normalizados
- Versão final após 7 iterações de debugging (v1→v7)

### Passo 3: Treino Exp A (equatorial) — **FEITO**
- Run v7_normalized: $\alpha_1$ erro 0.002%, data loss $1.95 \times 10^{-7}$
- 4 fases: warmup → $\alpha_1$-only → refinamento → L-BFGS
- Resultados em `results/equatorial/proton1/expA_v7_normalized/`

### Passo 4: Experimentos B/C (equatorial) — PENDENTE
- Esparsidade (30%, 5%) e ruído ($\sigma = 0.01, 0.05, 0.1$)
- Prótons 2 e 3

### Passo 5: Extensão para 3D — PENDENTE
- Gerar dados com `sv_3d.c` (`generate_dataset_3d.py`)
- O modelo já suporta `mode="3d"` (3 outputs: $\rho, Z, \varphi$; 3 residuais)
- Criar `train_inverse_3d.py` com mesma receita de 4 fases
- Escolha de $n_\text{frequencies}$ depende das oscilações nos dados 3D

---

## 12. Referências de código existente

- **PINN inversa na esfera**: `problems/stormer-problem/nn/pinn-inverse-issue5/pinn_stormer_inverse.py` — estrutura de referência original (first-order, com $\tau$-scaling)
- **Solver equatorial**: `problems/stormer-problem/simulation/no_constraint_case/equatorial_case/sv_equatorial.c` — NÃO MODIFICAR
- **Solver 3D**: `problems/stormer-problem/simulation/no_constraint_case/3d_case/sv_3d.c` — NÃO MODIFICAR
- **Relatório**: `references/stormer-problem/relatorio_final_pibic_pedro.pdf` — Seções 3.3.1 (equatorial) e 3.3.2 (3D)

---

## 13. Resumo das iterações de debugging (equatorial)

| Versão | Formulação | Resultado | Causa do problema |
|--------|------------|-----------|-------------------|
| v1 | 1ª ordem ($\rho, w_\rho, \varphi$), $\tau$-scaling, joint Adam | $\alpha_1$ erro 85-90% | $w_\rho$ não observável, joint Adam cria mínimos espúrios |
| v2 | v1 + kinematic consistency no warmup | Data loss degradou para 0.024 | Competição de objetivos no warmup |
| v3 | v1 + ODE weight ramp-up | $\alpha_1$ erro 11% | Desestabiliza quando peso chega a 1.0 |
| v4 | 2ª ordem ($\rho, \varphi$), joint Adam | $\alpha_1$ erro 7-90% | Joint Adam ainda cria mínimos espúrios |
| v5 | 2ª ordem + $\alpha_1$-only phase | $\alpha_1$ erro <0.1% | **Conceito correto** — primeiro sucesso na convergência de $\alpha_1$ |
| v6 | v5 M-free, sem normalização | $\alpha_1$ erro 0.076%, trajetória flat | ODE loss $\sim 10^9$ destruiu o data fit na Fase 2 |
| **v7** | **v5 M-free + normalização de residuais** | **$\alpha_1$ erro 0.002%, trajetória perfeita** | **Versão final** |
