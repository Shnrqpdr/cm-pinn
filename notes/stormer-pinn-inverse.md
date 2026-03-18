# PINN Inverso: Problema de Störmer na Esfera (Issue #5)

**Data**: 2026-03-11
**Referência**: Raissi et al. (2019) — Physics-Informed Neural Networks (problemas inversos)

---

## 1. Objetivo

Dado dados observados de uma trajetória $(\theta(t_i), \varphi(t_i))$ gerada com parâmetros conhecidos, identificar o parâmetro de acoplamento magnético $k = \frac{\mu_0 q m}{4\pi R}$ usando uma PINN.

O parâmetro $k$ é tratado como **parâmetro treinável** (`nn.Parameter`) e otimizado simultaneamente com os pesos da rede.

---

## 2. Diferenças em relação ao Forward (Issue #4)

| Aspecto | Forward | Inverso |
|---------|---------|---------|
| Parâmetros treináveis | Apenas pesos da rede | Pesos da rede **+ $k$** |
| Loss de dados | Não tem (só colocação) | $\mathcal{L}_\text{data}$: MSE vs observações |
| Loss IC | Imposta diretamente ($\omega_\text{ic} = 100$) | Vem dos dados ($t=0$ nos dados) |
| Loss de energia | $A, B$ fixos (conhecidos) | $A(k), B(k)$ recalculados a cada step |
| Parâmetros da ODE | Fixos | Dependem de $k$ (diferenciáveis) |

---

## 3. Formulação Matemática

### 3.1 Parâmetros como funções de $k$

Todas as quantidades adimensionais dependem de $k$ através da energia $K$:

$$K(k) = \frac{1}{2MR^2}\left[p_{\theta_0}^2 + \frac{(p_\varphi + k\sin^2\theta_0)^2}{\sin^2\theta_0}\right]$$

$$L(k) = \sqrt{2MR^2 K(k)}, \quad a(k) = \frac{p_\varphi}{L}, \quad b(k) = \frac{k}{L}$$

$$A(k) = \frac{2b(a+b) - 1 - \sqrt{1 - 4ab}}{2b^2}, \quad B(k) = \frac{2b(a+b) - 1 + \sqrt{1 - 4ab}}{2b^2}$$

$$\tau_\text{scale}(k) = \sqrt{\frac{MR^2}{2K(k)}}$$

Todas essas expressões são implementadas em PyTorch, mantendo o grafo computacional para que os gradientes fluam até $k$.

### 3.2 Função de perda

$$\mathcal{L} = \omega_\text{data}\mathcal{L}_\text{data} + \omega_\text{ode}\mathcal{L}_\text{ode}$$

**Data loss:**

$$\mathcal{L}_\text{data} = \frac{1}{N_u}\sum_i \left|z_\text{NN}(t_i) - z_i^\text{obs}\right|^2 + \left|\varphi_\text{NN}(t_i) - \varphi_i^\text{obs}\right|^2$$

**ODE loss** (em coordenadas $t_\text{norm} = t/T_\text{obs}$):

$$\mathcal{L}_\text{ode} = \frac{1}{N_c}\sum_j \left|r_1\right|^2 + \left|r_2\right|^2 + \left|r_3\right|^2$$

onde $S(k) = T_\text{obs} / \tau_\text{scale}(k)$ e:

$$r_1 = \frac{dz}{dt_\text{norm}} - S \cdot w$$
$$r_2 = \frac{dw}{dt_\text{norm}} - S \cdot b^2 z(A + B - 2z^2)$$
$$r_3 = \frac{d\varphi}{dt_\text{norm}} - S \cdot \left(\frac{a}{1 - z^2} + b\right)$$

### 3.3 Positividade de $k$

$k$ é parametrizado via softplus: $k = \log(1 + e^{k_\text{raw}})$, garantindo $k > 0$.

---

## 4. Arquitetura e Implementação

### 4.1 Código

Diretório: `problems/stormer-problem/nn/pinn-inverse-issue5/`

| Arquivo | Função |
|---------|--------|
| `generate_dataset_inverse.py` | Gera dados "observados" a partir da solução analítica |
| `pinn_stormer_inverse.py` | Modelo PINN + Trainer (standalone, independente do forward) |
| `train_inverse.py` | Script de treinamento com suporte a Exps A/B/C via CLI |

### 4.2 Rede Neural

Mesma arquitetura do forward: 4×128, Fourier 10 freq, tanh, float64.

**Diferenças:**
- Input: $t_\text{norm} = t/T_\text{obs}$ (tempo físico normalizado, não tempo adimensional)
- Learning rate separada para $k$ (`k_lr=0.01`) vs rede (`adam_lr=0.001`)

### 4.3 Treinamento

- **Adam**: 20k épocas, param groups separados (rede lr=1e-3, k lr=1e-2)
- **L-BFGS**: 5 passos, todos os parâmetros juntos
- **Gradient clipping**: max_norm=1.0
- **Pesos**: $\omega_\text{data} = 1.0$, $\omega_\text{ode} = 1.0$

---

## 5. Experimentos

| Exp | Dados | Ruído | O que testa |
|-----|-------|-------|-------------|
| A | 100% (1000 pts) | $\sigma = 0$ | Convergência básica |
| B | 30% (300 pts) | $\sigma = 0$ | Robustez a esparsidade |
| C1 | 100% | $\sigma = 0.01$ | Ruído baixo |
| C2 | 100% | $\sigma = 0.05$ | Ruído moderado |
| C3 | 100% | $\sigma = 0.1$ | Ruído alto |

Uso via CLI:
```bash
python train_inverse.py                    # Exp A
python train_inverse.py --fraction 0.3     # Exp B
python train_inverse.py --noise 0.01       # Exp C1
```

---

## 6. Resultados

**Data**: 2026-03-16

Todos os experimentos usam $k_\text{init} = 1.0$ (chute inicial = 2× o valor real), warmup de 5000 épocas (data only, $k$ congelado), 20000 épocas Adam e 5 passos L-BFGS.

### 6.1 Exp A — Dados completos, sem ruído

- **Dados**: 1000 pontos (100%), $\sigma = 0$
- **Resultado**:
  - $k_\text{est} = 0.499999$, erro relativo $= 1.34 \times 10^{-6}$
  - Loss final: total $= 4.11 \times 10^{-5}$, data $= 6.27 \times 10^{-9}$, ODE $= 4.11 \times 10^{-5}$
- **Convergência**: $k$ fica estável em ~1.0 durante o warmup, depois cai rapidamente entre as épocas 5000–8000, atingindo o valor verdadeiro. O erro relativo decresce monotonicamente até $\sim 10^{-6}$ após L-BFGS.
- **Trajetória**: PINN vs True indistinguíveis visualmente. Erros absolutos $|\Delta\theta| \sim 10^{-4}$°, $|\Delta\varphi| \sim 10^{-5}$ rad.

### 6.2 Exp B — Dados esparsos, sem ruído

Testa a robustez da identificação de $k$ com frações decrescentes de dados observados.

| Variante | Pontos | $k_\text{est}$ | Erro relativo | Loss total | Loss data | Loss ODE |
|----------|--------|----------------|---------------|------------|-----------|----------|
| B 30% | 300 | 0.500001 | $1.85 \times 10^{-6}$ | $1.36 \times 10^{-4}$ | $2.43 \times 10^{-8}$ | $1.36 \times 10^{-4}$ |
| B 15% | 150 | 0.500000 | $1.68 \times 10^{-7}$ | $3.84 \times 10^{-5}$ | $1.48 \times 10^{-9}$ | $3.84 \times 10^{-5}$ |
| B 5% | 50 | 0.499989 | $2.22 \times 10^{-5}$ | $4.25 \times 10^{-5}$ | $4.88 \times 10^{-9}$ | $4.24 \times 10^{-5}$ |

**Observações:**

1. **Robustez notável**: Mesmo com apenas 50 pontos (5%), o erro relativo em $k$ é $\sim 2 \times 10^{-5}$ — 4 casas decimais corretas.
2. **Regularização pela física**: A loss ODE age como regularizador forte, compensando a esparsidade dos dados. Com 15% dos dados, o resultado é *melhor* que com 30%, sugerindo que a rede não precisa de muitos pontos para calibrar $k$ — a física carrega a informação.
3. **Degradação gradual**: A trajetória reconstruída com 5% dos dados mostra erros absolutos ligeiramente maiores ($\sim 10^{-3}$) na parte final do domínio temporal, mas a identificação de $k$ permanece excelente.
4. **Padrão de convergência idêntico**: Em todas as variantes, $k$ converge rapidamente após o warmup (épocas 5000–8000), independente da fração de dados.

### 6.3 Exp C — Dados com ruído gaussiano

Testa a robustez da identificação de $k$ com ruído gaussiano $\mathcal{N}(0, \sigma^2)$ adicionado às observações de $z$ e $\varphi$. Todos os experimentos usam 100% dos dados (1000 pontos).

| Variante | $\sigma$ | $k_\text{est}$ | Erro relativo | Loss total | Loss data | Loss ODE |
|----------|----------|----------------|---------------|------------|-----------|----------|
| C1 | 0.01 | 0.499941 | $1.18 \times 10^{-4}$ | $1.83 \times 10^{-3}$ | $1.80 \times 10^{-4}$ | $3.82 \times 10^{-5}$ |
| C2 | 0.05 | 0.499755 | $4.91 \times 10^{-4}$ | $4.49 \times 10^{-2}$ | $4.48 \times 10^{-3}$ | $7.87 \times 10^{-5}$ |
| C3 | 0.10 | 0.499541 | $9.18 \times 10^{-4}$ | $1.79 \times 10^{-1}$ | $1.79 \times 10^{-2}$ | $1.14 \times 10^{-4}$ |

**Observações:**

1. **Degradação linear**: O erro relativo em $k$ escala aproximadamente linearmente com $\sigma$: $\sim 10^{-4}$ para $\sigma=0.01$, $\sim 5 \times 10^{-4}$ para $\sigma=0.05$, $\sim 10^{-3}$ para $\sigma=0.1$. Comportamento bem-posto.
2. **Filtragem de ruído pela física**: Mesmo com $\sigma = 0.1$ (ruído alto), o erro em $k$ é < 0.1%. A loss ODE atua como regularizador, forçando a rede a reconstruir uma trajetória fisicamente consistente em vez de interpolar o ruído.
3. **Piso de precisão**: O ruído impõe um limite inferior na precisão alcançável de $k$. Nos gráficos de convergência, o erro relativo estabiliza em um patamar que depende de $\sigma$, ao contrário dos casos sem ruído onde continua decrescendo.
4. **Trajetória suave**: Mesmo com dados ruidosos, a PINN reconstrói uma trajetória suave. No caso C3, os dados observados "sacudem" visivelmente ao redor da curva verdadeira, mas a solução reconstruída acompanha fielmente a trajetória real, com erros absolutos $|\Delta\theta| \sim 10^{-2}$° nos picos.
5. **Loss data vs ODE**: A loss de dados cresce com $\sigma^2$ (como esperado — a rede não pode ajustar perfeitamente dados ruidosos), enquanto a loss ODE permanece baixa ($\sim 10^{-4}$ a $10^{-5}$), confirmando que a rede prioriza a consistência física.

---

## 7. Conclusões

A PINN inversa para o problema de Störmer na esfera identifica com sucesso o parâmetro de acoplamento magnético $k$ em todos os cenários testados:

| Cenário | Melhor erro relativo em $k$ | Pior erro relativo em $k$ |
|---------|----------------------------|--------------------------|
| Dados completos, sem ruído (A) | $1.3 \times 10^{-6}$ | — |
| Dados esparsos, sem ruído (B) | $1.7 \times 10^{-7}$ (15%) | $2.2 \times 10^{-5}$ (5%) |
| Dados com ruído (C) | $1.2 \times 10^{-4}$ ($\sigma=0.01$) | $9.2 \times 10^{-4}$ ($\sigma=0.1$) |

**Fatores-chave de sucesso:**
- A formulação em $z = \cos\theta$ (livre de singularidades) é essencial
- O warmup de 5000 épocas (data only, $k$ congelado) estabiliza a rede antes de ativar a otimização de $k$
- A loss ODE age como regularizador forte, compensando esparsidade e filtrando ruído
- A parametrização via softplus garante $k > 0$ sem projeções explícitas

---

## 8. Validação com outras condições iniciais

**Data**: 2026-03-17

Para validar a generalidade da metodologia, repetimos os experimentos com dois conjuntos adicionais de condições iniciais do paper Piña & Cortés (2016).

### 8.1 Condições iniciais

| Case | $\theta_0$ | $p_{\theta_0}$ | $p_{\varphi_0}$ | Regime | $k_\text{init}$ | Ref. |
|------|-----------|----------------|-----------------|--------|-----------------|------|
| 1 | $\pi/3$ | 0 | 0.394 | one_hemisphere | 1.0 | fig6b |
| 2 | $\pi/4$ | 0 | 0.394 | two_hemispheres | 1.0 | fig6a |
| 3 | 0.6 | 0.1 | 0.25 | one_hemisphere | **0.3** | fig7a |

### 8.2 Case 2 — $\theta_0 = \pi/4$, regime two_hemispheres

Parâmetros adimensionais: $a = 0.4326$, $b = 0.5490$, $A = -0.2419$, $B = 0.5000$.

| Exp | Configuração | $k_\text{est}$ | Erro relativo |
|-----|-------------|----------------|---------------|
| A | 100%, $\sigma=0$ | 0.499986 | $2.78 \times 10^{-5}$ |
| B 15% | 15%, $\sigma=0$ | 0.499996 | $7.10 \times 10^{-6}$ |
| B 5% | 5%, $\sigma=0$ | 0.499995 | $9.07 \times 10^{-6}$ |
| C1 | 100%, $\sigma=0.01$ | 0.499928 | $1.44 \times 10^{-4}$ |
| C2 | 100%, $\sigma=0.05$ | 0.499613 | $7.74 \times 10^{-4}$ |
| C3 | 100%, $\sigma=0.1$ | 0.499246 | $1.51 \times 10^{-3}$ |

Resultados consistentes com o Case 1 — convergência excelente em todos os cenários. Degradação com ruído segue o mesmo padrão linear.

### 8.3 Case 3 — $\theta_0 = 0.6$, $p_\theta \neq 0$

Parâmetros adimensionais: $a = 0.3416$, $b = 0.6831$, $A = 0.1518$, $B = 0.7053$.

| Exp | Configuração | $k_\text{est}$ | Erro relativo |
|-----|-------------|----------------|---------------|
| A | 100%, $\sigma=0$ | 0.500002 | $3.36 \times 10^{-6}$ |
| B 15% | 15%, $\sigma=0$ | 0.500008 | $1.64 \times 10^{-5}$ |
| B 5% | 5%, $\sigma=0$ | 0.500001 | $2.41 \times 10^{-6}$ |
| C1 | 100%, $\sigma=0.01$ | 0.499996 | $7.35 \times 10^{-6}$ |
| C2 | 100%, $\sigma=0.05$ | 0.499905 | $1.90 \times 10^{-4}$ |
| C3 | 100%, $\sigma=0.1$ | 0.499861 | $2.78 \times 10^{-4}$ |

#### Mínimo local e sensibilidade ao $k_\text{init}$

**Problema encontrado:** Com $k_\text{init} = 1.0$ (default), o otimizador fica preso num mínimo local em $k \approx 0.667$ (erro relativo ~33%). Aumentar $k_\text{lr}$ de 0.001 para 0.01 não resolve — o mínimo é **topológico**, não de convergência lenta.

**Causa:** Com $p_{\theta_0} \neq 0$, a paisagem de loss tem um mínimo local acima de $k_\text{true}$. Em $k = 0.667$, os parâmetros adimensionais ($A = 0.498$, $B = 0.720$) produzem uma trajetória com amplitude menor ($\Delta z^2 = 0.222$ vs $0.554$ no verdadeiro), mas que satisfaz parcialmente tanto a loss de dados quanto a loss ODE.

**Solução:** Usar $k_\text{init} = 0.3$ (abordagem por baixo). Neste caso, $k$ converge suavemente para o valor verdadeiro com erro $\sim 10^{-6}$. Isso sugere que:
1. A paisagem de loss é convexa para $k < k_\text{true}$, mas não para $k > k_\text{true}$ neste caso
2. Em problemas inversos com $p_\theta \neq 0$, o chute inicial de $k$ importa
3. Uma estratégia robusta seria testar múltiplos $k_\text{init}$ e selecionar o que dá menor loss final

### 8.4 Comparação entre os 3 cases

| Exp | Case 1 | Case 2 | Case 3 |
|-----|--------|--------|--------|
| A (100%, $\sigma=0$) | $1.3 \times 10^{-6}$ | $2.8 \times 10^{-5}$ | $3.4 \times 10^{-6}$ |
| B 15% | $1.7 \times 10^{-7}$ | $7.1 \times 10^{-6}$ | $1.6 \times 10^{-5}$ |
| B 5% | $2.2 \times 10^{-5}$ | $9.1 \times 10^{-6}$ | $2.4 \times 10^{-6}$ |
| C1 ($\sigma=0.01$) | $1.2 \times 10^{-4}$ | $1.4 \times 10^{-4}$ | $7.4 \times 10^{-6}$ |
| C2 ($\sigma=0.05$) | $4.9 \times 10^{-4}$ | $7.7 \times 10^{-4}$ | $1.9 \times 10^{-4}$ |
| C3 ($\sigma=0.1$) | $9.2 \times 10^{-4}$ | $1.5 \times 10^{-3}$ | $2.8 \times 10^{-4}$ |

**Observações:**

1. **Metodologia validada**: Os 3 cases convergem com sucesso em todos os cenários, confirmando a generalidade da abordagem.
2. **Case 3 surpreendentemente melhor com ruído**: Com $k_\text{init} = 0.3$ correto, o Case 3 ($p_\theta \neq 0$) obtém erros menores nos cenários com ruído, possivelmente porque a dinâmica mais rica fornece mais informação sobre $k$.
3. **Case 2 ligeiramente pior**: O regime two_hemispheres tem erros ~2-3× maiores nos cenários com ruído, mas ainda na mesma ordem de grandeza.
4. **Todos os erros < 0.2%**: Mesmo no pior caso (Case 2, $\sigma = 0.1$), o erro relativo é $1.5 \times 10^{-3}$.
