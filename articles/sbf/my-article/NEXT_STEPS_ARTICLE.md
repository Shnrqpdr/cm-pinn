# Próximos passos para completar o artigo

## 1. ~~Gerar a figura de validação (caso esférico)~~ ✅ Concluído (2026-03-22)

Geradas 3 figuras separadas, cada uma com trajetória 3D sobre a esfera + erro de energia ΔH/H₀:
- `validation_sphere_banda.pdf` — banda no hemisfério norte (t=3000)
- `validation_sphere_equador.pdf` — travessia do equador (t=3000)
- `validation_sphere_loops.pdf` — laços/loops (t=5000)

Script: `make_validation_sphere.py`. Integrador C (`sv_sphere.c`) atualizado com saída `%.15e` e cálculo de H no arquivo de fase.


## 2. ~~Figuras existentes — verificar resolução~~ ✅ Concluído (2026-03-22)

- `Potencial_Efetivo_equatorial.png` — regenerada em 300 dpi (1883×1402px)
- `plotCom3Protons.pdf` — OK (PDF vetorial)
- `plot3DPlano2.pdf` — OK (PDF vetorial)


## 3. Considerar figuras adicionais (opcionais)

### 3a. Trajetórias 3D na esfera
Os casos do artigo de Cortés (2015) que já temos simulados em `run.sh`:
- Fig 6(a): theta0=pi/4, p_theta0=0, p_phi=0.394 (banda horizontal)
- Fig 6(b): theta0=pi/3, p_theta0=0, p_phi=0.394 (banda mais larga)
- Fig 6(c): theta0=75°, p_theta0=0, p_phi=-0.394 (loops)
- Fig 7(a): theta0=0.6, p_theta0=0.1, p_phi=0.25 (hemisfério norte)
- Fig 7(c): theta0=0.6, p_theta0=0.2525, p_phi=0.25 (cruza equador)

Selecionar 2-3 desses para uma figura mostrando a variedade de trajetórias na esfera.

### 3b. Espaço de fase
O relatório original inclui figuras do espaço de fase (rho, drho) para o caso 3D livre, mostrando a transição de órbitas regulares para caóticas. Considerar incluir no artigo.


## 4. ~~Verificar passo temporal do caso 3D~~ ✅ Concluído (2026-03-22)

Artigo atualizado para $\Delta t = 0.0002$ (valor no código `sv_3d.c`).


## Correções já realizadas (2026-03-22)

- [x] Expressão de $V_{\text{max}}$: corrigido de $c_{20}^2/(32m^2\alpha_1^2/c_{20}^2)$ para $c_{20}^4/(32m^4\alpha_1^2)$
- [x] Valores de $M_z$, $\alpha$ e $\alpha_1$: atualizados para consistência com o código ($M_z \approx 8.2 \times 10^{15}$, $\alpha_1 \approx 3.04 \times 10^3$ para prótons)
- [x] Abstract em inglês: atualizado para incluir o caso esférico
- [x] Passo temporal 3D: alterado de 0.00001 para 0.0002 (valor no código)
- [x] Notação $m$ na seção da esfera: corrigido para $\mu$ (momento de dipolo), evitando colisão com $m$ (massa)
- [x] Explicação de $c_{20}$ e $c_2$: adicionada fórmula e unidades
- [x] Roadmap (Seção 1.3): atualizado para mencionar os três casos (equatorial, 3D, esfera)
- [x] Passada humanizer: eliminadas expressões vagas, trailing participles, copula avoidance
- [x] Figuras de validação esférica: 3 figuras separadas (banda, equador, laços) com trajetória + erro de energia
- [x] Resolução do potencial efetivo equatorial: regenerada em 300 dpi
- [x] Propriedades simpléticas: seção detalhada com referências (Hairer et al.)
- [x] `sv_sphere.c`: saída numérica com precisão total (%.15e) e cálculo de H interno
