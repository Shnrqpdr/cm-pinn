# PINN Forward: Problema de Störmer restrito à esfera (Issue #4)

**Data**: 2026-03-08
**Referência**: Piña & Cortés, *Eur. J. Phys.* **37** (2016) 065009 — *"Störmer problem restricted to a spherical surface and the Euler and Lagrange tops"*

---

## 1. Descrição do Problema

Partícula carregada confinada a uma superfície esférica de raio $R$, sob a ação de um dipolo magnético no centro da esfera. O Hamiltoniano em coordenadas esféricas $(\theta, \varphi)$ é:

$$H = \frac{1}{2MR^2}\left[p_\theta^2 + \frac{(p_\varphi + k\sin^2\theta)^2}{\sin^2\theta}\right]$$

onde $M$ é a massa, $k = \mu_0 q m / (4\pi R)$ é o acoplamento magnético, e $p_\varphi$ é constante do movimento (coordenada cíclica).

### Parâmetros físicos utilizados

| Parâmetro | Valor |
|-----------|-------|
| $M$ (massa) | 2.0 |
| $R$ (raio) | 10.0 |
| $k$ (acoplamento magnético) | 0.5 |

### Adimensionalização

Definindo $L = \sqrt{2MR^2 K}$ (momento característico) e $\tau = t/\tau_\text{scale}$ com $\tau_\text{scale} = \sqrt{MR^2/(2K)}$:

$$a = \frac{p_\varphi}{L}, \quad b = \frac{k}{L}$$

A substituição $z = \cos\theta$ transforma o problema no sistema:

$$\dot{z}^2 = b^2(B - z^2)(z^2 - A)$$

onde $A$ e $B$ são raízes do polinômio quártico $V(z) = \frac{b^2}{2}(z^2-A)(z^2-B)$, com $A + B = 1 + 2ab + a^2$ e $AB = a^2$.

A solução analítica usa **funções elípticas de Jacobi**: $\text{dn}$ para regime de um hemisfério ($A > 0$), $\text{cn}$ para cruzamento do equador ($A < 0$).

---

## 2. Solução Analítica — Implementação e Validação

### 2.1 Implementação

Arquivo: `problems/stormer-problem/analytical/stormer_sphere_analytical.py`

A solução analítica foi implementada em Python usando `scipy.special.ellipj` para as funções elípticas de Jacobi e `scipy.integrate.cumulative_trapezoid` para a integração numérica de $\varphi(t)$:

$$\frac{d\varphi}{d\tau} = \frac{a}{1 - z^2} + b$$

Para condições iniciais arbitrárias (com $p_\theta \neq 0$), foi necessário computar um offset de fase $u_0$ no argumento das funções elípticas, resolvido via inversão por bisseção.

### 2.2 Validação contra integrador Störmer-Verlet

Arquivo: `problems/stormer-problem/analytical/validate_analytical.py`

A solução analítica foi comparada com o integrador simplético em C (`sv_sphere.c`, $\Delta t = 0.0002$) para 4 casos distintos:

| Caso | Regime | $\theta$ erro médio | $\varphi$ erro médio | Resíduo ODE |
|------|--------|---------------------|----------------------|-------------|
| fig6b (um hemisfério) | $A > 0$ | ~1e-3 | ~1e-3 | ~1e-14 |
| fig6c (loops) | $A < 0$ | ~1e-3 | ~1e-3 | ~1e-14 |
| fig7a (hemisfério, $p_\theta \neq 0$) | $A > 0$ | ~1e-3 | ~1e-3 | ~1e-14 |
| fig7c (cruza equador) | $A < 0$ | ~1e-3 | ~1e-3 | ~1e-14 |

O resíduo ODE ($\dot{z}^2$ vs $b^2(B-z^2)(z^2-A)$) ficou na ordem de $10^{-14}$ (precisão de máquina), confirmando a correção da solução analítica. O erro ~$10^{-3}$ na comparação com Störmer-Verlet é consistente com o erro do integrador numérico ($\Delta t = 0.0002$).

---

## 3. PINN — Tentativas e Evolução

### 3.1 Tentativa 1: Equações de Hamilton em tempo físico

**Formulação**: rede neural $(\theta, p_\theta, \varphi) = \text{NN}(t)$ com as equações de Hamilton originais:

$$\dot{\theta} = \frac{p_\theta}{MR^2}, \quad \dot{p}_\theta = \frac{(p_\varphi + k\sin^2\theta)\cos\theta}{MR^2\sin^3\theta}\left[(p_\varphi + k\sin^2\theta) - 2k\sin^2\theta\right]$$

**Resultado**: Loss ficou estagnada em ~2300, validação com $\theta_\text{MAE} = 3.6$ rad (essencialmente aleatório).

**Causa raiz**: O fator de escala $1/T_\text{final} \approx 7 \times 10^{-5}$ (com $T_\text{final} \approx 5507$) suprimia os gradientes da ODE, tornando a perda de condição inicial dominante e a perda física irrelevante.

### 3.2 Tentativa 2: Formulação adimensional em $\theta$

**Formulação**: tempo adimensional $\tau_\text{norm} \in [0, 1]$, Fourier features, rede prediz $(\theta, p_\theta, \varphi)$ com ODEs adimensionais:

$$\frac{d\theta}{d\tau} = -\frac{p_\theta}{L\sin\theta}, \quad \frac{dp_\theta}{d\tau} = \frac{\cos\theta}{L\sin^3\theta}\left[(a + \sin^2\theta)((a + \sin^2\theta) - 2\sin^2\theta)\right]$$

**Resultado**: Loss explodiu para NaN imediatamente (loss ODE inicial: $1.3 \times 10^{13}$).

**Causa raiz**: O termo $1/\sin^3\theta$ é singular em $\theta = 0$ e $\theta = \pi$. Durante o treinamento, os pesos da rede inicialmente produzem valores arbitrários de $\theta$, incluindo valores próximos dos polos, causando explosão numérica.

### 3.3 Tentativa 3 (final): Formulação em $z = \cos\theta$ — Convergência

**Formulação**: substituição $z = \cos\theta$, $w = dz/d\tau$, eliminando completamente a singularidade:

$$\frac{dz}{d\tau} = w$$
$$\frac{dw}{d\tau} = b^2 z(A + B - 2z^2)$$
$$\frac{d\varphi}{d\tau} = \frac{a}{1 - z^2} + b$$

**Por que funciona**:
- Todos os coeficientes da ODE são $O(1)$ — não há fator de escala suprimindo gradientes
- Não há singularidade em $z = \pm 1$ (os polos correspondem a $1 - z^2 = 0$, mas a trajetória fisica nunca alcança os polos para $A > 0$)
- O pequeno epsilon ($10^{-8}$) no denominador de $\varphi$ previne instabilidade numérica
- Integral de energia $w^2 = b^2(B-z^2)(z^2-A)$ serve como constraint adicional

---

## 4. Arquitetura Final da PINN

### 4.1 Rede Neural

Arquivo: `problems/stormer-problem/nn/pinn_stormer.py`

| Componente | Especificação |
|------------|---------------|
| Entrada | $\tau_\text{norm} \in [0, 1]$ (tempo adimensional normalizado) |
| Fourier features | $[\tau, \sin(2\pi f_i \tau), \cos(2\pi f_i \tau)]$, $f_i \in \{1, ..., 10\}$ → dim = 21 |
| Camadas ocultas | 4 camadas, 128 neurônios cada |
| Ativação | $\tanh$ |
| Saída | 3 valores: $(z, w, \varphi)$ |
| Inicialização | Xavier normal (pesos), zeros (bias) |
| Precisão | `float64` |
| Total de parâmetros | ~51.000 |

### 4.2 Função de Perda

$$\mathcal{L} = \omega_\text{ode} \mathcal{L}_\text{ode} + \omega_\text{ic} \mathcal{L}_\text{ic} + \omega_\text{energy} \mathcal{L}_\text{energy}$$

- $\mathcal{L}_\text{ode}$: resíduo médio quadrático das 3 ODEs nos pontos de colocação (3000 pontos LHS)
- $\mathcal{L}_\text{ic}$: erro quadrático nas condições iniciais ($z_0, w_0, \varphi_0$) em $\tau = 0$
- $\mathcal{L}_\text{energy}$: violação da integral de energia $|w^2 - b^2(B-z^2)(z^2-A)|^2$

| Peso | Valor | Justificativa |
|------|-------|---------------|
| $\omega_\text{ode}$ | 1.0 | Base |
| $\omega_\text{ic}$ | 100.0 | Prioriza satisfação das condições iniciais |
| $\omega_\text{energy}$ | 10.0 | Constraint de conservação como regularizador |

### 4.3 Estratégia de Treinamento

Arquivo: `problems/stormer-problem/nn/train.py`

**Fase 1 — Adam** (20.000 épocas):
- Learning rate inicial: $10^{-3}$
- Scheduler: Cosine annealing ($\eta_\text{min} = 10^{-6}$)
- Gradient clipping: $\|\nabla\|_\text{max} = 1.0$

**Fase 2 — L-BFGS** (5 passos externos):
- `max_iter=20`, `max_eval=25` por passo
- `history_size=100`
- Line search: Strong Wolfe
- Tolerâncias: `grad=1e-12`, `change=1e-14`

### 4.4 Dados

Arquivo de geração: `problems/stormer-problem/nn/generate_dataset.py`
Dataset: `problems/stormer-problem/nn/data/dataset_case1_one_hemisphere.npz`

| Conjunto | Pontos | Método |
|----------|--------|--------|
| Referência (plotting) | 10.000 | Uniforme |
| Colocação (treinamento) | 3.000 | Latin Hypercube Sampling |
| Validação | 2.000 | Uniforme (aleatório) |

**Caso 1** (um hemisfério): $\theta_0 = \pi/3$, $p_{\theta_0} = 0$, $\varphi_0 = 0$, $p_\varphi = 0.394$
- Regime: $A > 0$ (partícula não cruza o equador)
- 2 períodos de oscilação em $\theta$
- $T_\text{final} \approx 5506.55$ (tempo físico)

---

## 5. Resultados — Caso 1 (Um Hemisfério)

### 5.1 Convergência do Treinamento

![Histórico de loss](../problems/stormer-problem/nn/results/case1/loss_history.png)

A loss total converge suavemente durante a fase Adam, com descida de ~$10^{-1}$ para ~$10^{-6}$. A fase L-BFGS proporciona refinamento adicional até ~$3 \times 10^{-7}$. As componentes individuais mostram que:
- A loss IC ($\mathcal{L}_\text{ic}$) converge primeiro (peso alto $\omega_\text{ic} = 100$), atingindo $10^{-17}$
- A loss de energia segue, atingindo $10^{-12}$
- A loss ODE domina o total final

### 5.2 Comparação PINN vs Solução Analítica

![PINN vs Analítica](../problems/stormer-problem/nn/results/case1/pinn_vs_analytical.png)

As curvas da PINN e da solução analítica são visualmente indistinguíveis para $\theta(t)$ e $\varphi(t)$. O painel de erros mostra que o erro absoluto permanece na faixa $10^{-4}$ a $10^{-3}$ ao longo de toda a trajetória.

### 5.3 Conservação de Energia

![Resíduo da integral de energia](../problems/stormer-problem/nn/results/case1/energy_conservation.png)

O resíduo da integral de energia $|w^2 - b^2(B-z^2)(z^2-A)|$ permanece na ordem de $10^{-6}$ a $10^{-7}$, confirmando que a PINN aprendeu a conservar a energia do sistema.

### 5.4 Trajetória 3D

![Trajetória 3D](../problems/stormer-problem/nn/results/case1/trajectory_3d.png)

A trajetória na esfera é reproduzida com fidelidade. A partícula oscila no hemisfério superior ($\theta < \pi/2$), consistente com o regime $A > 0$.

### 5.5 Métricas de Treinamento

| Métrica | Valor |
|---------|-------|
| Loss total | $3.19 \times 10^{-7}$ |
| Loss ODE | $3.19 \times 10^{-7}$ |
| Loss IC | $2.25 \times 10^{-17}$ |
| Loss energia | $1.10 \times 10^{-12}$ |

### 5.6 Métricas de Validação

| Métrica | $\theta$ | $\varphi$ |
|---------|----------|-----------|
| MAE | $1.01 \times 10^{-4}$ rad | $1.31 \times 10^{-4}$ rad |
| Erro máximo | $3.37 \times 10^{-4}$ rad | $7.71 \times 10^{-4}$ rad |
| L2 relativo | $1.20 \times 10^{-4}$ | $1.26 \times 10^{-5}$ |

---

## 6. Lições Aprendidas

1. **Escolha de coordenadas é crítica para PINNs**: singularidades nas equações ($1/\sin^3\theta$) impedem convergência. A transformação $z = \cos\theta$ eliminou a singularidade e tornou todos os coeficientes $O(1)$.

2. **Adimensionalização é essencial**: trabalhar em tempo físico ($T_\text{final} \sim 5500$) suprime gradientes da física. O tempo adimensional normalizado $\tau_\text{norm} \in [0, 1]$ equilibra as escalas.

3. **Integral de energia como regularizador**: incluir $w^2 = b^2(B-z^2)(z^2-A)$ como termo de loss adicional melhora a conservação de energia e a qualidade geral da solução.

4. **Fourier features para soluções periódicas**: o embedding $[\tau, \sin(2\pi f_i \tau), \cos(2\pi f_i \tau)]$ com 10 frequências captura eficientemente o comportamento oscilatório.

5. **Adam + L-BFGS é uma combinação robusta**: Adam faz a descida grossa com robustez, L-BFGS refina com convergência superlinear. Mas L-BFGS com muitas iterações externas em plateau é desperdício.

6. **Gradient clipping previne instabilidade**: essencial nos estágios iniciais quando os Fourier features podem produzir gradientes grandes.

7. **float64 é necessário**: precisão dupla evita perda de significância nas derivadas automáticas e nas tolerâncias do L-BFGS.
