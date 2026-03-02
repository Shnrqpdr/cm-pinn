# Physics-Informed Neural Networks (PINNs) - Guia Completo

> Documento de referência consolidando o conhecimento das referências do repositório e o estado da arte em PINNs. Este documento serve como guia principal para o desenvolvimento dos modelos neste projeto.

---

## 1. Fundamentos e Motivação

### 1.1 O que são PINNs

Physics-Informed Neural Networks (PINNs) são redes neurais que incorporam leis físicas — tipicamente descritas por equações diferenciais parciais (EDPs) ou ordinárias (EDOs) — diretamente na função de perda durante o treinamento. A ideia central é que a rede neural não apenas ajuste dados observados, mas simultaneamente satisfaça as equações governantes do sistema físico.

**Referência fundacional:** Raissi, Perdikaris & Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations", Journal of Computational Physics, 2019.

### 1.2 Motivações centrais

1. **Escassez de dados:** Em muitos sistemas físicos, a aquisição de dados é cara ou proibitiva. PINNs compensam a falta de dados usando as equações governantes como regularizador, permitindo aprendizado eficiente com poucos exemplos.

2. **Método livre de malha (mesh-free):** Diferente de métodos numéricos clássicos (FEM, diferenças finitas, volumes finitos), PINNs não requerem discretização do domínio em malhas. A solução é avaliada em pontos de colocação amostrados no domínio.

3. **Problemas diretos e inversos no mesmo framework:** O mesmo código pode resolver:
   - **Problemas diretos (forward):** dadas as equações e condições de contorno/iniciais, encontrar a solução $u(t,x)$.
   - **Problemas inversos:** dados observações esparsas da solução, identificar parâmetros desconhecidos $\lambda$ da EDP.

4. **Unificação de dados e física:** PINNs operam como um framework de aprendizado multi-tarefa onde a rede deve simultaneamente ajustar dados observados e satisfazer resíduos da EDP.

### 1.3 Contexto histórico

- **Precursores (1990s):** Dissanayake & Phan-Thien (1994) e Lagaris et al. (1998) já propunham redes neurais para resolver EDPs, usando redes rasas com poucas camadas.
- **Processos Gaussianos (2015-2017):** Raissi & Karniadakis usaram regressão por processos Gaussianos para incorporar informação física, mas com limitações em problemas não-lineares.
- **PINNs modernas (2017-2019):** Raissi et al. introduziram o framework PINN usando redes profundas com diferenciação automática, superando limitações anteriores.
- **Explosão de citações (2019+):** O número de papers sobre PINNs quintuplicou entre 2019-2020 e dobrou novamente em 2021.

---

## 2. Formulação Matemática

### 2.1 Formulação geral

Considere uma EDP da forma geral:

$$\frac{\partial u}{\partial t} + \mathcal{N}[u; \lambda] = 0, \quad x \in \Omega, \quad t \in [0, T]$$

onde:
- $u(t, x)$ é a solução latente (desconhecida)
- $\mathcal{N}[u; \lambda]$ é um operador diferencial não-linear com parâmetros $\lambda$
- $\Omega \subset \mathbb{R}^D$ é o domínio espacial
- Condições de contorno: $B(u, x, t) = 0$ em $\partial\Omega$
- Condições iniciais: $u(x, 0) = u_0(x)$

### 2.2 Construção da PINN

Define-se o **resíduo da EDP**:

$$f(t, x) := \frac{\partial u}{\partial t} + \mathcal{N}[u; \lambda]$$

Uma rede neural profunda $u_{\text{NN}}(t, x; \Theta)$ aproxima a solução $u(t,x)$. Aplicando diferenciação automática (autograd) para computar as derivadas parciais de $u_{\text{NN}}$ em relação a $t$ e $x$, obtém-se a rede informada pela física $f_{\text{NN}}(t, x; \Theta)$.

**Ponto crucial:** Ambas as redes — $u_{\text{NN}}$ e $f_{\text{NN}}$ — compartilham os mesmos parâmetros $\Theta$ (pesos e bias). A rede $f_{\text{NN}}$ não tem parâmetros treináveis adicionais; ela é construída puramente por diferenciação automática sobre $u_{\text{NN}}$.

### 2.3 Função de perda

A função de perda composta tem a forma:

$$L(\Theta) = \omega_{\text{data}} \cdot L_{\text{data}} + \omega_{\text{pde}} \cdot L_{\text{pde}} + \omega_{\text{bc}} \cdot L_{\text{bc}} + \omega_{\text{ic}} \cdot L_{\text{ic}}$$

onde:

- **$L_{\text{data}}$** (dados): MSE entre predições e dados observados (se disponíveis)
  $$L_{\text{data}} = \frac{1}{N_u} \sum_{i=1}^{N_u} |u_{\text{NN}}(t_i, x_i; \Theta) - u_i|^2$$

- **$L_{\text{pde}}$** (resíduo da EDP): MSE do resíduo nos pontos de colocação
  $$L_{\text{pde}} = \frac{1}{N_f} \sum_{i=1}^{N_f} |f_{\text{NN}}(t_i, x_i; \Theta)|^2$$

- **$L_{\text{bc}}$** (condições de contorno):
  $$L_{\text{bc}} = \frac{1}{N_b} \sum_{i=1}^{N_b} |B(u_{\text{NN}}) - g|^2$$

- **$L_{\text{ic}}$** (condições iniciais):
  $$L_{\text{ic}} = \frac{1}{N_0} \sum_{i=1}^{N_0} |u_{\text{NN}}(0, x_i; \Theta) - u_0(x_i)|^2$$

Os pesos $\omega$ controlam a importância relativa de cada termo. O balanceamento desses pesos é **crítico** para a convergência (ver Seção 5).

### 2.4 Pontos de colocação

Os pontos de colocação $\{t_i, x_i\}$ onde o resíduo da EDP é avaliado são tipicamente gerados por:

- **Latin Hypercube Sampling (LHS):** Estratégia quasi-aleatória que garante cobertura uniforme do domínio. Método mais usado na literatura de PINNs.
- **Distribuição uniforme:** Mais simples, mas pode ter cobertura pior em altas dimensões.
- **Amostragem adaptativa:** Concentra pontos em regiões de alto resíduo (técnica mais avançada).

### 2.5 Condições de contorno: Soft vs. Hard enforcement

**Soft enforcement (vanilla PINN):**
As condições de contorno são incluídas como termos na função de perda. Simples de implementar, mas as condições são satisfeitas apenas aproximadamente.

**Hard enforcement (PCNN - Physics-Constrained NN):**
A arquitetura da rede é modificada para que a saída automaticamente satisfaça as condições de contorno para qualquer entrada. Por exemplo, para condições de Dirichlet homogêneas $u(0)=u(1)=0$:

$$u_{\text{hard}}(x) = x(1-x) \cdot u_{\text{NN}}(x)$$

Vantagem: Simplifica o problema de otimização, eliminando termos da função de perda.

---

## 3. Modelos de Tempo Contínuo vs. Discreto

### 3.1 Modelo de tempo contínuo

O modelo contínuo trata tempo e espaço simetricamente como entradas da rede:

- **Entrada:** $(t, x_1, x_2, \ldots, x_d)$
- **Saída:** $u(t, x)$
- **Treinamento:** Minimiza o resíduo da EDP em pontos de colocação espalhados por todo o domínio espaço-temporal.

**Vantagens:** Predição contínua em qualquer ponto $(t,x)$; formulação simples.
**Desvantagens:** Pode ter dificuldade com domínios temporais longos; necessita muitos pontos de colocação em altas dimensões.

### 3.2 Modelo de tempo discreto (Runge-Kutta)

Proposto por Raissi et al. (2019) como alternativa que elimina a necessidade de pontos de colocação temporais.

**Ideia:** Usar um esquema Runge-Kutta implícito com $q$ estágios para avançar de $t_n$ para $t_{n+1}$:

$$u^{n+c_j} = u^n + \Delta t \sum_{j=1}^{q} a_{ij} g[u^{n+c_j}], \quad i = 1, \ldots, q$$
$$u^{n+1} = u^n + \Delta t \sum_{j=1}^{q} b_j g[u^{n+c_j}]$$

A rede neural prediz $[u^{n+c_1}, \ldots, u^{n+c_q}, u^{n+1}]$ a partir da entrada espacial $x$. O esquema é então "invertido": o lado direito (dependente da rede) é usado para estimar $u^n$, que é comparado com os dados conhecidos em $t_n$.

**Vantagens notáveis:**
- O número de estágios $q$ pode ser muito grande (ex: $q=100$) sem aumento significativo do custo computacional, pois adicionar um estágio apenas acrescenta um neurônio à camada de saída.
- Esquemas Gauss-Legendre implícitos permanecem A-estáveis independentemente da ordem, ideais para problemas rígidos (stiff).
- Permite passos temporais muito grandes mantendo estabilidade e precisão.

**Exemplo marcante:** No paper original, a equação de Allen-Cahn foi resolvida com $q=100$ estágios em um único passo temporal ($\Delta t=0.8$), com erro temporal teórico de $O(\Delta t^{200}) \approx 10^{-20}$.

---

## 4. Arquiteturas de Rede Neural

### 4.1 Feed-Forward Neural Network (FFNN / MLP)

A arquitetura padrão das PINNs. Uma rede fully-connected com $L$ camadas:

$$u_{\text{NN}}(x) = \sigma_L \circ W_L \circ \sigma_{L-1} \circ W_{L-1} \circ \cdots \circ \sigma_1 \circ W_1 (x)$$

**Configurações típicas na literatura:**
- Raissi et al. (2019): 5 camadas, 100 neurônios/camada; ou 9 camadas, 20 neurônios/camada
- HFM (Hidden Fluid Mechanics): 10 camadas, 300 neurônios/camada (50 por variável de saída × 6 variáveis)
- Tartakovsky et al.: 3 camadas ocultas, 50 neurônios/camada

**Múltiplas redes:** Para sistemas acoplados (ex: Navier-Stokes), pode-se usar uma única rede com múltiplas saídas ou múltiplas redes compartilhando informação.

### 4.2 Funções de ativação

A escolha da função de ativação é crítica em PINNs porque derivadas de segunda (ou maior) ordem da saída da rede são computadas via autograd. A função de ativação deve ser suficientemente suave ($k+1$ vezes diferenciável para EDP de ordem $k$).

| Função | Fórmula | Propriedades |
|--------|---------|-------------|
| **tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Mais usada em PINNs; $C^\infty$; simétrica; pode sofrer vanishing gradient |
| **sin** | $\sin(x)$ | Usada no HFM; $C^\infty$; estável numericamente para derivadas altas |
| **Swish** | $x \cdot \sigma(\beta x)$ | $\beta$ treinável; supera tanh em convergência segundo alguns estudos |
| **SiLU** | $x \cdot \sigma(x)$ | Swish com $\beta=1$; usada em PINNs causais |
| **ReLU** | $\max(0, x)$ | **Evitar em PINNs** — segunda derivada é zero em toda parte |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $C^\infty$ mas não simétrica; usada menos frequentemente |

**Recomendação:** Para problemas gerais, `tanh` é a escolha padrão segura. Para problemas que requerem derivadas de alta ordem ou soluções oscilatórias, `sin` pode ser superior.

### 4.3 Outras arquiteturas

- **CNN (Convolutional):** Usadas em PhyGeoNet para domínios com geometria complexa, transformando o domínio irregular em domínio retangular de referência via transformação de coordenadas.
- **RNN (Recurrent):** Naturais para dependência temporal; usadas em RNN-DCT-PINN para melhor tratamento de EDPs dependentes do tempo.
- **Encoder-Decoder:** Usados para modelar fluxos estocásticos e redução de dimensionalidade.
- **DeepONet:** Usa duas sub-redes (branch net + trunk net) para aprender operadores. Especialmente genérica — sem restrições na topologia das sub-redes.
- **Redes Rasas / ELM:** X-TFC usa uma única camada treinada pelo algoritmo Extreme Learning Machine, com custo computacional menor.

### 4.4 Redes modificadas para PINNs

- **Modified MLP (Wang et al.):** Adiciona transformações $U = \sigma(W_u \cdot x + b_u)$ e $V = \sigma(W_v \cdot x + b_v)$ e modifica cada camada como: $H^l = (1 - Z^l) \odot U + Z^l \odot V$, onde $Z^l$ é a saída da camada $l$. Melhora significativamente o fluxo de gradiente.
- **Fourier Feature Networks:** Mapeiam entradas para features de Fourier antes da rede, ajudando a superar o spectral bias.

---

## 5. Treinamento: Otimizadores e Estratégias

### 5.1 Otimizadores

**Estratégia padrão (Adam + L-BFGS):**
A combinação mais robusta empiricamente observada na literatura:

1. **Fase 1 — Adam:** Otimizador de primeira ordem com taxa de aprendizado adaptativa. Usado nas primeiras $N$ épocas (ex: 3000-10000) para chegar a uma vizinhança do mínimo.
   - Learning rate típico: $10^{-3}$ a $10^{-4}$
   - Decay exponencial recomendado

2. **Fase 2 — L-BFGS:** Otimizador quasi-Newton de segunda ordem. Usado após Adam para refinamento final com convergência rápida local.
   - Full-batch (sem mini-batches)
   - Muito eficiente quando próximo do mínimo

**Variação no HFM (Raissi et al., 2018):** Treinamento progressivo com learning rates decrescentes: 250 epochs com $\text{lr}=10^{-3}$, depois 500 com $\text{lr}=10^{-4}$, depois 250 com $\text{lr}=10^{-5}$.

### 5.2 Balanceamento da função de perda (Loss Balancing)

**Este é um dos problemas mais críticos das PINNs.** Os diferentes termos da função de perda (dados, EDP, contorno, inicial) tipicamente têm magnitudes muito diferentes, causando:
- O termo de resíduo da EDP frequentemente domina, obscurecendo as condições de contorno/iniciais.
- A rede pode satisfazer a EDP de maneira trivial (ex: $u \equiv 0$) em vez da solução correta.

**Técnicas de balanceamento:**

1. **Pesos fixos empíricos:** Multiplicar termos menores por fatores escalares determinados empiricamente. Ex: Kollmannsberger et al. usaram fator $5 \times 10^{-4}$ no termo $\text{MSE}_f$.

2. **Learning Rate Annealing (Wang et al., 2021):** Ajusta dinamicamente os pesos usando as estatísticas dos gradientes de cada termo.

3. **NTK-based weighting:** Usa a teoria de Neural Tangent Kernel para balancear as taxas de convergência dos diferentes termos da perda.

4. **Self-Adaptive Weights:** Trata os pesos como parâmetros treináveis adicionais.

5. **GradNorm:** Normaliza os gradientes de cada termo da perda para terem magnitude similar.

### 5.3 Treinamento causal

Para EDPs dependentes do tempo, treinamento padrão pode falhar porque a rede tenta satisfazer a física em tempos futuros antes de resolver tempos anteriores.

**Causal Training (Wang et al.):** Divide o domínio temporal em segmentos e pondera a perda acumulativamente — regiões temporais posteriores só contribuem significativamente quando regiões anteriores já convergem.

Demonstrado no paper de PINNs para a equação de Schrödinger: sem treinamento causal, o erro explode em domínios temporais longos (MSE $\sim 10^{-2}$); com treinamento causal, o erro cai para $\sim 10^{-3}$.

### 5.4 Amostragem adaptativa de pontos de colocação

Em vez de fixar os pontos de colocação, redistribuí-los durante o treinamento:

- **Residual-based Adaptive Refinement (RAR):** Adiciona novos pontos de colocação nas regiões de maior resíduo.
- **Adaptive sampling:** Usa a distribuição do resíduo como função de probabilidade para amostrar novos pontos.

---

## 6. Análise de Erros e Teoria

### 6.1 Decomposição do erro total

O erro total de uma PINN pode ser decomposto em (De Ryck & Mishra, Acta Numerica 2024):

$$\text{Erro Total} = \text{Erro de Aproximação} + \text{Erro de Generalização} + \text{Erro de Treinamento}$$

- **Erro de aproximação:** Quão bem a classe de redes neurais pode representar a solução verdadeira. Depende da arquitetura (largura, profundidade).
- **Erro de generalização:** Diferença entre o risco empírico (nos pontos de treinamento) e o risco populacional (em todo o domínio). Diminui com o número de pontos de colocação.
- **Erro de treinamento:** Quão bem o otimizador encontra o mínimo do risco empírico. **Identificado como o gargalo principal** na performance de PINNs.

### 6.2 Erro de aproximação

Redes neurais são **aproximadores universais** (Hornik et al., 1989): qualquer função contínua pode ser aproximada arbitrariamente bem por um MLP com uma camada oculta e neurônios suficientes.

Para PINNs especificamente:
- Redes com ativação tanh de profundidade $O(\log(1/\varepsilon))$ e largura $O(1/\varepsilon^d)$ podem aproximar funções suaves com erro $\varepsilon$ em dimensão $d$.
- Redes mais profundas são exponencialmente mais eficientes que redes rasas para certas classes de funções.

### 6.3 Estabilidade e papel na análise de erros

A **estabilidade** da EDP subjacente é fundamental para a análise de erros de PINNs:

- Se a EDP é bem-posta (well-posed) — i.e., pequenas perturbações nos dados produzem pequenas perturbações na solução — então minimizar o resíduo da EDP garante convergência para a solução verdadeira.
- Para problemas inversos (data assimilation), a estabilidade é condicional e depende de resultados como desigualdades de observabilidade.

**Implicação prática:** PINNs funcionam melhor para EDPs bem-postas e com soluções regulares. Problemas com soluções descontínuas (choques), multi-escala ou caóticas são significativamente mais difíceis.

### 6.4 Erro de generalização

O número de pontos de colocação $N_f$ necessários para boa generalização depende da dimensão $d$ do problema e da regularidade da solução:

- Para soluções suaves: $N_f \sim O(1/\varepsilon^d)$ (maldição da dimensionalidade para métodos clássicos)
- PINNs podem potencialmente superar a maldição da dimensionalidade para certas classes de EDPs, embora resultados teóricos completos ainda não estejam disponíveis.

### 6.5 Erro de treinamento — o gargalo

O erro de treinamento é o **principal limitante prático** das PINNs:

- A função de perda de PINNs é **altamente não-convexa** com paisagem de perdas complexa.
- Otimizadores baseados em gradiente podem ficar presos em mínimos locais ou pontos de sela.
- O balanceamento entre os diferentes termos da perda afeta criticamente o treinamento.
- **Spectral bias:** Redes neurais tendem a aprender primeiro componentes de baixa frequência, dificultando a captura de soluções com características de alta frequência.

---

## 7. Problemas Conhecidos e Limitações

### 7.1 Spectral Bias

Redes neurais treinadas por gradiente descendente aprendem preferenciamente componentes de baixa frequência. Isso é problemático para:
- Soluções oscilatórias (equação de Schrödinger)
- Soluções com múltiplas escalas
- Interfaces afiadas ou descontinuidades

**Mitigações:**
- Fourier Feature Networks: mapear entradas $x \to [\sin(\omega x), \cos(\omega x)]$ com múltiplas frequências $\omega$
- Escalamento de variáveis (adimensionalização cuidadosa)
- Redes multi-escala

### 7.2 Convergência para soluções triviais

PINNs podem convergir para soluções triviais (ex: $u \equiv 0$) que satisfazem a EDP homogênea mas violam condições de contorno. Isso ocorre quando:
- Os pesos da perda de contorno/inicial são insuficientes
- O resíduo da EDP domina o treinamento

### 7.3 Domínios temporais longos

PINNs contínuas tendem a perder acurácia para tempos longos porque:
- O domínio espaço-temporal se torna grande demais
- A rede não respeita naturalmente a causalidade temporal

**Soluções:**
- Treinamento causal (Seção 5.3)
- Time-stepping: resolver sequencialmente em janelas temporais
- Modelo discreto de Runge-Kutta (Seção 3.2)

### 7.4 Alta dimensionalidade

Embora PINNs tenham potencial para superar a maldição da dimensionalidade (pois são mesh-free), na prática o número de pontos de colocação necessários cresce rapidamente com a dimensão.

### 7.5 Problemas rígidos (stiff)

Sistemas com múltiplas escalas temporais (stiff) são desafiadores. O modelo discreto com Runge-Kutta implícito oferece vantagem aqui.

### 7.6 Descontinuidades e choques

PINNs baseadas em MLP produzem soluções suaves por construção (funções de ativação suaves). Capturar descontinuidades requer técnicas especiais:
- Decomposição de domínio
- Funções de ativação adaptativas
- Conservative PINNs (CPINNs)

---

## 8. Variantes e Extensões

### 8.1 Variational PINNs (VPINNs / hp-VPINNs)

Em vez de usar o resíduo forte (pointwise), usa-se a formulação variacional/fraca da EDP:

$$L_{\text{vpinn}} = \int f_{\text{NN}}(x) \cdot v(x) \, dx$$

onde $v(x)$ são funções teste. Pode ser mais robusta para problemas com soluções menos regulares.

### 8.2 Conservative PINNs (CPINNs)

Projetadas para preservar leis de conservação. Decompõem o domínio em subdomínios e impõem continuidade e conservação de fluxo nas interfaces.

### 8.3 Physics-Constrained Neural Networks (PCNNs)

Impõem condições de contorno/iniciais rigidamente (hard BC) na arquitetura da rede, ao invés de na função de perda (soft BC). O treinamento precisa minimizar apenas o resíduo da EDP.

### 8.4 Domain Decomposition Methods

Dividem o domínio em subdomínios, cada um com sua própria rede PINN, com condições de interface para acoplar as soluções.

### 8.5 DeepONet e Neural Operators

Aprendem o **operador** solução $\mathcal{G}: f \to u$, não uma solução particular. Uma vez treinado, pode resolver a EDP para diferentes condições iniciais/contorno sem re-treinamento. Conceito mais poderoso, mas requer mais dados de treinamento.

### 8.6 Transfer Learning para PINNs

Pré-treinar a PINN em um problema mais simples (ou com parâmetros diferentes) e transferir os pesos para o problema alvo. Pode acelerar significativamente a convergência.

---

## 9. Aplicações Demonstradas nas Referências

### 9.1 Mecânica dos fluidos

**Raissi et al. (2019) — Paper original:**
- Equação de Burgers (1D, não-linear)
- Equação de Schrödinger não-linear (valores complexos)
- Equação de Allen-Cahn (reação-difusão)
- Equações de Korteweg-de Vries (ondas em águas rasas)

**Raissi et al. (2018) — Hidden Fluid Mechanics (HFM):**
- Navier-Stokes 2D e 3D: inferência de campos de velocidade e pressão a partir de visualização de escalares passivos (fumaça/corante).
- Fluxo em torno de cilindros ($\text{Re}=100$, $\text{Re}=200$)
- Fluxo 3D de esteira de cilindro com Navier-Stokes
- Demonstrou capacidade de inferir pressão sem dados diretos de pressão
- Aplicação biomédica: potencial para hemodinâmica em artérias

**Problema direto (Navier-Stokes):**
- Aprendeu os parâmetros $\lambda_1$ (convecção) e $\lambda_2$ (viscosidade) com erros $< 1\%$ e $< 6\%$, respectivamente
- Robusto a 1% de ruído gaussiano nos dados

### 9.2 Mecânica quântica

**Shah et al. — PINNs para equação de Schrödinger dependente do tempo:**
- Oscilador harmônico quântico unidimensional
- Superposição de autoestados $|\psi_{01}\rangle$ e $|\psi_{03}\rangle$
- Decomposição da função de onda complexa $\psi = u + iv$ em parte real e imaginária
- Resultados do baseline: MSE $\sim 10^{-5}$ para $|\psi|^2$
- **Generalização:** Treinou para $\omega \in [0.75, 2.0]$ e testou para $\omega \in [0.5, 2.5]$ — boa interpolação, extrapolação degradada
- **Domínios temporais longos:** Sem tratamento causal, erro explode (MSE $\sim 10^{-2}$); com treinamento causal, erro cai para $\sim 10^{-3}$
- **Estados de alta energia:** PINN padrão converge falsamente para estado fundamental; treinamento causal resolve o problema

**Detalhes de implementação:**
- 6 camadas, 512 neurônios cada (FCN-PINN)
- Ativação: tanh (baseline) ou SiLU (causal)
- Otimizador: Adam com $\beta_1=0.09$, $\beta_2=0.999$, $\text{lr}=10^{-3}$ com decay exponencial

### 9.3 Nano-óptica e metamateriais

**Chen et al. — PINNs para problemas inversos em nano-óptica:**
- Recuperação de parâmetros efetivos de permissividade em sistemas de espalhamento com nanoestruturas
- Validação contra FEM (Finite Element Method)
- Demonstrou potencial para design de metamateriais considerando efeitos de radiação e tamanho finito

### 9.4 Descoberta de equações governantes

**Chen, Liu & Sun — Descoberta de EDPs a partir de dados esparsos:**
- Framework combinando: representação por rede neural profunda + embedding físico + regressão esparsa
- Passos: (1) aproximar solução com NN, (2) computar derivadas via autograd, (3) identificar termos dominantes via regressão esparsa
- Testado em: equação de Burgers, Korteweg-de Vries, reação-difusão, Navier-Stokes
- Funciona com dados ruidosos e esparsos

### 9.5 Transferência de calor

**Kollmannsberger et al. — Exemplos pedagógicos:**
- Barra estática linear elástica (EDO 1D)
- Equação de calor não-linear 1D (transiente)
- Modelo contínuo e discreto comparados
- Implementação detalhada em PyTorch com código

---

## 10. Implementação Prática

### 10.1 Frameworks e bibliotecas

| Framework | Linguagem | Observações |
|-----------|-----------|-------------|
| **DeepXDE** | Python (TensorFlow/PyTorch/JAX) | Framework mais usado para PINNs; API de alto nível; suporte a muitas EDPs |
| **NVIDIA Modulus** | Python (PyTorch) | Framework industrial; GPU-otimizado; antigo SimNet |
| **NeuralPDE.jl** | Julia (Flux) | Integrado ao ecossistema SciML de Julia |
| **NeuroDiffEq** | Python (PyTorch) | Foco em EDOs e EDPs simples |
| **PyTorch direto** | Python | Flexibilidade total; recomendado para pesquisa |
| **JAX direto** | Python | Diferenciação automática eficiente; composabilidade; JIT compilation |

### 10.2 Estrutura típica de implementação em PyTorch

```python
import torch
import torch.nn as nn
from torch.autograd import grad

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # Construir rede MLP
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)  # Última camada sem ativação

    def compute_pde_residual(self, t, x):
        """Computa o resíduo da EDP via autograd"""
        t.requires_grad_(True)
        x.requires_grad_(True)

        inputs = torch.cat([t, x], dim=1)
        u = self.forward(inputs)

        # Derivadas parciais via autograd
        u_t = grad(u, t, grad_outputs=torch.ones_like(u),
                    create_graph=True, retain_graph=True)[0]
        u_x = grad(u, x, grad_outputs=torch.ones_like(u),
                    create_graph=True, retain_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                     create_graph=True, retain_graph=True)[0]

        # Resíduo da EDP (ex: equação de calor u_t = α * u_xx)
        f = u_t - alpha * u_xx
        return f

# Treinamento
def train(model, t_data, x_data, u_data, t_colloc, x_colloc, epochs):
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer_adam.zero_grad()

        # Loss dos dados
        u_pred = model(torch.cat([t_data, x_data], dim=1))
        loss_data = torch.mean((u_pred - u_data)**2)

        # Loss da EDP
        f_pred = model.compute_pde_residual(t_colloc, x_colloc)
        loss_pde = torch.mean(f_pred**2)

        # Loss total (com pesos)
        loss = loss_data + weight_pde * loss_pde

        loss.backward()
        optimizer_adam.step()

    # Fase L-BFGS
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters())
    # ... (implementação L-BFGS com closure)
```

### 10.3 Checklist de implementação

1. **Adimensionalizar as equações** — Normalizar variáveis para que entradas e saídas estejam em escalas similares (idealmente $[0,1]$ ou $[-1,1]$).

2. **Escolher arquitetura** — Começar com: 4-8 camadas ocultas, 20-100 neurônios/camada, ativação tanh.

3. **Gerar pontos de colocação** — LHS é o padrão; usar $N_f \gg N_{\text{data}}$.

4. **Inicializar pesos** — Xavier/Glorot initialization é o padrão para tanh.

5. **Configurar treinamento** — Adam ($10^{-3} \to 10^{-4}$) + L-BFGS.

6. **Monitorar cada componente da loss** separadamente — Verificar se nenhum termo domina excessivamente.

7. **Validar contra soluções analíticas** (se disponíveis) ou soluções numéricas de referência.

### 10.4 Dicas de depuração

- Se a loss não diminui: reduzir learning rate, verificar escalamento, simplificar o problema
- Se $L_{\text{pde}}$ diminui mas $L_{\text{bc}}/L_{\text{ic}}$ não: aumentar pesos dos termos de contorno
- Se a solução é trivial ($u \approx 0$): aumentar significativamente o peso das condições de contorno/iniciais
- Se há oscilações: verificar derivadas com autograd, tentar ativação sin em vez de tanh
- Para domínios temporais longos: usar treinamento causal ou time-stepping

---

## 11. PINNs para Mecânica Clássica (Relevância para este Projeto)

### 11.1 Formulação Lagrangiana

Para um sistema mecânico com Lagrangiana $L = T - V$ (energia cinética menos potencial):

As equações de Euler-Lagrange:

$$\frac{d}{dt} \left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = 0$$

podem ser incorporadas diretamente na loss da PINN. A rede neural parametriza a trajetória $q(t)$ e o resíduo das equações de Euler-Lagrange é minimizado.

### 11.2 Formulação Hamiltoniana

Para um sistema com Hamiltoniano $H(q, p)$:

As equações de Hamilton:

$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}$$

A PINN pode parametrizar $(q(t), p(t))$ ou diretamente $H(q, p)$ como rede neural (Hamiltonian Neural Networks).

### 11.3 Leis de conservação

PINNs podem ser treinadas com termos adicionais na perda para impor conservação:
- **Energia:** $\frac{dH}{dt} = 0$ para sistemas conservativos
- **Momento angular:** $\frac{dL}{dt} = 0$ na ausência de torques
- **Momento linear:** $\frac{dp}{dt} = 0$ na ausência de forças externas

Alternativamente, usar arquiteturas que preservem essas simetrias por construção (Symplectic Neural Networks, Hamiltonian Neural Networks).

### 11.4 Sistemas de EDOs

Para problemas em mecânica clássica, frequentemente lidamos com sistemas de EDOs (não EDPs). A formulação PINN se simplifica:

- Entrada: tempo $t$
- Saída: coordenadas generalizadas $q_1(t), q_2(t), \ldots, q_n(t)$
- Loss da EDP: resíduo das equações de movimento
- Condições iniciais: $q(0) = q_0$, $\dot{q}(0) = v_0$

A ausência de dimensões espaciais torna o treinamento significativamente mais rápido e estável que para EDPs.

### 11.5 Problema de Störmer (relevância direta)

O problema de Störmer — movimento de uma partícula carregada no campo magnético de um dipolo — é um sistema de EDOs de segunda ordem que pode ser formulado como:

1. Escrever as equações de movimento em coordenadas generalizadas
2. Parametrizar a trajetória $(r(t), \theta(t), \phi(t))$ como saídas da PINN
3. O resíduo inclui as equações de Euler-Lagrange do sistema
4. Condições iniciais definem posição e velocidade inicial da partícula

As leis de conservação (energia e momento angular azimuthal) podem ser incorporadas como termos adicionais na perda ou impostas rigidamente via arquitetura da rede.

---

## 12. Direções Futuras e Questões Abertas

### 12.1 Questões teóricas

- Garantias de convergência formais para o treinamento de PINNs
- Taxas de convergência ótimas em relação ao número de pontos de colocação
- Análise rigorosa do spectral bias e como mitigá-lo
- Condições sob as quais PINNs superam métodos numéricos clássicos

### 12.2 Questões práticas

- Seleção automática de hiperparâmetros (arquitetura, pesos da loss)
- Balanceamento robusto e automático da função de perda
- Escalabilidade para problemas 3D complexos e sistemas acoplados
- Quantificação de incerteza (Bayesian PINNs)
- Integração com dados experimentais ruidosos
- Treinamento distribuído e paralelismo

### 12.3 Direções promissoras

- **Neural Operators:** Aprender operadores em vez de soluções pontuais
- **Multi-fidelity PINNs:** Combinar dados de diferentes níveis de fidelidade
- **PINNs para sistemas estocásticos:** Incorporar incerteza diretamente na formulação
- **Hybrid PINNs:** Combinar PINNs com métodos numéricos clássicos
- **Curricula de treinamento:** Estratégias progressivas (do simples ao complexo)

---

## 13. Referências do Repositório

| Arquivo | Conteúdo | Autores Principais |
|---------|----------|-------------------|
| `PINN-original.pdf` | Paper fundacional de PINNs | Raissi, Perdikaris, Karniadakis (2019) |
| `1808.04327v1.pdf` | Hidden Fluid Mechanics — Navier-Stokes | Raissi, Yazdani, Karniadakis (2018) |
| `2201.05624v4.pdf` | Survey: "Where we are and What's next" | Cuomo, Di Cola, Giampaolo, Rozza, Raissi, Piccialli (2022) |
| `Physics_Informed_Neural_Networks.pdf` | Capítulo de livro com exemplos pedagógicos em PyTorch | Kollmannsberger et al. (2021) |
| `BDCC-06-00140-v2.pdf` | Review sistemática e análise bibliométrica | Lawal et al. (2022) |
| `fa1a976cd67e1a1c222f16dbf2dd4545a886.pdf` | Review: "Progress and Challenges" | Antonion, Wang, Raissi, Joshie (2024) |
| `Physics informed neura networks as solver of the schrodinger equation.pdf` | PINNs para eq. Schrödinger dependente do tempo | Shah, Stiller, Hoffmann, Cangi |
| `Physics informed neural networks in nano-optics and metamaterials.pdf` | PINNs para problemas inversos em nano-óptica | Chen, Lu, Karniadakis, Dal Negro (2020) |
| `Physics informed neural networks with scarce data.pdf` | Descoberta de EDPs com dados esparsos | Chen, Liu, Sun |
| `numerical_analysis_of_physicsinformed_neural_networks...pdf` | Análise numérica rigorosa de PINNs | De Ryck, Mishra (Acta Numerica, 2024) |
