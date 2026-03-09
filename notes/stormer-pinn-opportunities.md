# Oportunidades de PINN para o Problema de Störmer

> Análise das possibilidades de aplicação de Physics-Informed Neural Networks ao problema de Störmer, baseada nos artigos em `references/stormer-problem/`.

---

## Contexto: O que dizem os artigos

### Artigo 1 — Störmer problem restricted to a spherical surface (Cortés & Poza)

O problema de Störmer restrito a uma superfície esférica é um sistema **integrável** com 2 graus de liberdade e 2 constantes de movimento:

- **Energia (hamiltoniano):**

$$H = \frac{1}{2MR^2}\left[p_\theta^2 + \frac{(p_\phi + k\sin^2\theta)^2}{\sin^2\theta}\right]$$

- **Momento canônico azimutal:** $p_\phi = [MR^2\dot{\phi} - k]\sin^2\theta = \text{const}$

O sistema reduz a um problema 1D com potencial efetivo do tipo **double-well**:

$$V_{\text{eff}}(\theta) = \frac{1}{2MR^2}(p_\phi + k\sin^2\theta)^2\csc^2\theta$$

Tipos de trajetória: confinamento em bandas horizontais, cruzamento de hemisférios, trajetórias com loops.

### Artigo 2 — Störmer problem and the Euler and Lagrange tops (Cortés & Poza)

Demonstra a **equivalência matemática exata** entre o problema de Störmer na esfera e os tops de Euler e Lagrange. Ambos os sistemas se reduzem a um **potencial quártico** em $z = \cos\theta$:

$$\dot{z}^2 + b^2z^4 + [1 - 2b(a+b)]z^2 + (a+b)^2 - 1 = 0$$

As soluções são expressas em **funções elípticas de Jacobi** (dn, cn, sn). A integral azimutal tem a mesma forma que a solução do top de Euler.

### Artigo 3 — Stochastic-Dissipative Störmer Problem (Harko et al., 2025)

Estende o problema clássico com **dissipação e forças estocásticas** (equação de Lorentz-Langevin):

$$m\frac{d^2\vec{r}}{dt^2} = q\vec{v} \times \vec{B} - \gamma m\vec{v} + m\vec{f}^{(s)}(t)$$

Três regimes: CSP (conservativo), CDSP (dissipativo), SDSP (estocástico-dissipativo). Resultados numéricos com o esquema de Milstein para integração estocástica. Análise de padrões de radiação e densidade espectral de potência (PSD).

### Artigo 4 — Dynamics on axisymmetric surface (abordagem geométrica)

Framework de geometria diferencial para partícula em superfície axissimétrica sob gravidade. Usa o referencial de Darboux $(\mathbf{T}, \mathbf{n}, \ell)$:

$$\mathbf{a} = a(t)\mathbf{T} + v^2(t)(\kappa_n \mathbf{n} + \kappa_g \ell)$$

Conservação de $L_3 = m\rho^2\dot{\phi}$ e energia. Potencial efetivo 1D. Aplica para cilindro (solução analítica) e catenoide (numérica).

---

## Recursos existentes

Simulações em C usando o método de Störmer-Verlet:
- `problems/stormer-problem/simulation/constraint_case/sphere/sv_sphere.c` — caso esférico ($\theta, \phi$)
- `problems/stormer-problem/simulation/no_constraint_case/3d_case/sv_3d.c` — caso 3D completo ($\rho, \phi, z$)
- `problems/stormer-problem/simulation/no_constraint_case/equatorial_case/sv_equatorial.c` — caso equatorial 2D

Esses simuladores podem gerar dados de treinamento/validação para as PINNs.

---

## Pergunta central: Como contribuir com PINN no problema de Störmer?

A resposta se desdobra em várias oportunidades, organizadas da mais direta à mais inovadora:

---

## Oportunidade 1: PINN Forward — Problema de Störmer na esfera

**Descrição:** Resolver o sistema de EDOs do problema de Störmer restrito à esfera usando PINN.

**Formulação:**
- Entrada: $t$
- Saída: $\theta(t), \phi(t)$
- Loss da EDP: resíduo das equações de Hamilton

$$\mathcal{L}_{\text{ode}} = \left|\ddot{\theta}_{\text{NN}} - f_\theta(\theta, \dot{\theta}, \dot{\phi})\right|^2 + \left|\ddot{\phi}_{\text{NN}} - f_\phi(\theta, \dot{\theta}, \dot{\phi})\right|^2$$

- Loss de conservação (soft constraint):

$$\mathcal{L}_{\text{energy}} = \left|\frac{dH}{dt}\right|^2, \quad \mathcal{L}_{p_\phi} = \left|\frac{dp_\phi}{dt}\right|^2$$

- Condições iniciais: $\theta(0), \dot{\theta}(0), \phi(0), \dot{\phi}(0)$

**Viabilidade:** Alta. Sistema 2D integrável, solução analítica conhecida (funções elípticas de Jacobi), dados do `sv_sphere.c` para validação. Excelente caso de benchmark.

**Contribuição:** Demonstrar que PINNs podem resolver problemas da mecânica clássica com campos magnéticos de dipolo de forma eficiente, comparando com integrador simplético.

---

## Oportunidade 2: PINN Forward — Problema de Störmer 3D completo

**Descrição:** Resolver o caso 3D não-restrito ($\rho, \phi, z$) com PINN.

**Formulação:**
- Entrada: $t$
- Saída: $\rho(t), \phi(t), z(t)$
- Loss da EDP: resíduo das equações de movimento 3D (derivadas do hamiltoniano)
- Loss de conservação: energia $H$ e momento angular azimutal $L_z$

**Desafio:** O sistema 3D é **não-integrável** e pode ser **caótico** (demonstrado por Ziglin-Yoshida). Requer:
- Treinamento causal (seção 5.3 do `pinns.md`)
- Time-stepping (janelas temporais)
- Possivelmente amostragem adaptativa

**Viabilidade:** Média-alta. Dados do `sv_3d.c` disponíveis. O caos torna o problema mais desafiador mas também mais interessante.

**Contribuição:** Investigar a capacidade de PINNs em sistemas caóticos da mecânica clássica — tema pouco explorado na literatura.

---

## Oportunidade 3: PINN Inverso — Identificação de parâmetros

**Descrição:** Dado dados de trajetória (gerados pelos simuladores C), usar PINN para identificar parâmetros físicos desconhecidos do sistema.

**Formulação:**
- Dados: trajetórias $(\theta(t_i), \phi(t_i))$ observadas (com possível ruído)
- Parâmetros a identificar: $k = \frac{q\mu_0 m}{4\pi R}$, massa $M$, raio $R$ (ou combinações)
- A PINN trata $k$ (ou $M$, $R$) como parâmetros treináveis junto com os pesos da rede

**Viabilidade:** Alta. É exatamente o tipo de problema que PINNs resolvem bem (ver paper original de Raissi et al. para Navier-Stokes). Dados abundantes dos simuladores.

**Contribuição:** Framework de data assimilation para inferir propriedades do dipolo magnético a partir de trajetórias observadas. Relevância para magnetosferas planetárias.

---

## Oportunidade 4: PINN para o problema Estocástico-Dissipativo

**Descrição:** Usar PINN para resolver a equação de Lorentz-Langevin com dissipação e ruído.

**Formulação:**
- Versão dissipativa (sem ruído):

$$\mathcal{L}_{\text{ode}} = \left|m\ddot{\vec{r}}_{\text{NN}} - q\dot{\vec{r}}_{\text{NN}} \times \vec{B} + \gamma m\dot{\vec{r}}_{\text{NN}}\right|^2$$

- Versão estocástica: PINNs para SDEs (Stochastic Differential Equations) — fronteira da pesquisa atual
  - Abordagem 1: PINN para a equação de Fokker-Planck associada (evolução da densidade de probabilidade)
  - Abordagem 2: PINN como surrogate model treinado com ensemble de trajetórias estocásticas

**Viabilidade:**
- Caso dissipativo: Média-alta (EDO determinística, apenas adiciona o termo de arraste)
- Caso estocástico: Média (PINNs para SDEs é área ativa de pesquisa, menos madura)

**Contribuição:** Muito alta potencialmente. O artigo de Harko et al. (2025) é recente e usa apenas métodos numéricos tradicionais (Milstein). Uma abordagem via PINN seria uma **contribuição original e contemporânea**.

---

## Oportunidade 5: Hamiltonian Neural Network (HNN) para o Störmer

**Descrição:** Em vez de resolver as equações de movimento diretamente, aprender o Hamiltoniano $H(q, p)$ como uma rede neural, e derivar as trajetórias a partir das equações de Hamilton.

**Formulação:**
- Rede neural: $H_{\text{NN}}(\theta, p_\theta, \phi, p_\phi)$
- As equações de movimento são derivadas automaticamente:

$$\dot{\theta} = \frac{\partial H_{\text{NN}}}{\partial p_\theta}, \quad \dot{p}_\theta = -\frac{\partial H_{\text{NN}}}{\partial \theta}$$

- A conservação de energia é **garantida por construção** (a rede aprende $H$ diretamente)
- Treinamento com dados de trajetória dos simuladores C

**Viabilidade:** Alta. HNNs são bem estabelecidas (Greydanus et al., 2019). O problema de Störmer é um excelente caso de teste.

**Contribuição:** Comparar HNN vs PINN clássica no problema de Störmer. Investigar se a estrutura hamiltoniana ajuda na captura de dinâmica caótica no caso 3D.

---

## Oportunidade 6: Neural Operator (DeepONet) — Mapeamento de condições iniciais para trajetórias

**Descrição:** Treinar um operador neural que mapeia condições iniciais diretamente para trajetórias completas, sem re-treinamento.

**Formulação:**
- Branch net: recebe condições iniciais $(\theta_0, \dot{\theta}_0, \phi_0, \dot{\phi}_0)$
- Trunk net: recebe tempo $t$
- Saída: $(\theta(t), \phi(t))$ para aquelas condições iniciais
- Treinamento: grande dataset de trajetórias geradas pelo `sv_sphere.c` e/ou `sv_3d.c`

**Viabilidade:** Média. Requer grande volume de dados (muitas trajetórias com diferentes condições iniciais). Os simuladores C podem gerar esses dados.

**Contribuição:** Exploração rápida do espaço de fases do problema de Störmer. Uma vez treinado, predição instantânea para qualquer condição inicial — útil para classificação de trajetórias (trapped, escaping, etc.).

---

## Oportunidade 7: Transfer Learning entre Störmer, Euler Top e Lagrange Top

**Descrição:** Explorar a equivalência matemática entre os três sistemas usando transfer learning de PINNs.

**Formulação:**
- Treinar PINN para o top de Euler (ou Lagrange) — problema clássico bem documentado
- Transferir pesos para o problema de Störmer na esfera (mesma estrutura de potencial quártico)
- Avaliar: aceleração de convergência, generalização entre domínios físicos

**Viabilidade:** Média. A equivalência é matemática (mesmas funções elípticas de Jacobi), mas as variáveis físicas são diferentes.

**Contribuição:** Muito original. Demonstraria que PINNs podem explorar conexões matemáticas profundas entre sistemas fisicamente distintos.

---

## Priorização sugerida

| # | Oportunidade | Viabilidade | Originalidade | Prioridade |
|---|-------------|-------------|---------------|------------|
| 1 | PINN Forward — esfera | Alta | Média | **1 (benchmark)** |
| 3 | PINN Inverso — parâmetros | Alta | Média-alta | **2** |
| 2 | PINN Forward — 3D completo | Média-alta | Alta | **3** |
| 5 | HNN para Störmer | Alta | Média-alta | **4** |
| 4 | PINN Estocástico-Dissipativo | Média | Muito alta | **5** |
| 6 | Neural Operator (DeepONet) | Média | Alta | **6** |
| 7 | Transfer Learning tops↔Störmer | Média | Muito alta | **7** |

A sugestão é começar pela oportunidade 1 como **benchmark/prova de conceito**, pois tem solução analítica conhecida e dados prontos. A partir daí, escalar para os casos mais complexos e originais.
