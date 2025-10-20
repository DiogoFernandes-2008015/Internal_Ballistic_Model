import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def internal_ballistics_model(t, y, params):
    """
    Define o sistema de EDOs para o modelo de balística interna
    de parâmetros concentrados.

    Estado y = [x, v, f]
      x: posição do projétil (m)
      v: velocidade do projétil (m/s)
      f: fração da alma do propelente queimada (adimensional, 0 a 1)

    params: um dicionário contendo todas as constantes do problema.
    """

    # Desempacotar o vetor de estado
    x, v, f = y

    # Desempacotar os parâmetros
    A = params['A']  # Área da base do projétil (m^2)
    m_proj = params['m_proj']  # Massa do projétil (kg)
    m_p0 = params['m_p0']  # Massa inicial do propelente (kg)
    rho_p = params['rho_p']  # Densidade do propelente sólido (kg/m^3)
    V_0 = params['V_0']  # Volume inicial da câmara (m^3)
    e_1 = params['e_1']  # Espessura da alma do propelente (m)
    beta = params['beta']  # Coeficiente da taxa de queima (m/s / Pa^alpha)
    alpha = params['alpha']  # Expoente da taxa de queima
    Lambda = params['Lambda']  # Força do propelente (J/kg)
    gamma = params['gamma']  # Razão dos calores específicos
    b = params['b']  # Covolume do gás (m^3/kg)
    theta = params['theta']  # Fator de forma do propelente

    # --- Cálculos Intermediários ---

    # 1. Massa efetiva (constante neste modelo simplificado)
    m_eff = m_proj + m_p0 / 2.0

    # 2. Fração de massa queimada (psi) a partir da fração de alma (f)
    # Função de forma para grão tubular: psi = (1-theta)*f + theta*f^2
    # Garantir que f não passe de 1
    f_calc = min(f, 1.0)
    psi = (1 - theta) * f_calc + theta * (f_calc ** 2)

    # 3. Massa de gás gerada
    m_g = m_p0 * psi

    # 4. Volume livre para o gás
    V_gas = V_0 + A * x - (m_p0 - m_g) / rho_p - m_g * b

    # Evitar divisão por zero ou volume negativo (deve ser positivo)
    if V_gas <= 0:
        # Isso indica um problema nos parâmetros (ex: V_0 muito pequeno)
        # Em um caso real, a ignição não ocorreria ou teríamos P infinito
        P = 1e9  # Simula uma pressão muito alta
    else:
        # 5. Calcular a Pressão (Equação de Energia)
        numerador = m_g * Lambda - (gamma - 1) / 2.0 * m_eff * v ** 2

        # A pressão não pode ser negativa
        if numerador <= 0:
            P = 0.0
        else:
            P = numerador / V_gas

    # Garantir que a pressão não seja negativa (fim da propulsão)
    P = max(0.0, P)

    # --- Calcular as Derivadas d(y)/dt ---

    # 1. dx/dt = v
    dxdt = v

    # 2. dv/dt = P * A / m_eff
    # (Adicionar aqui termos de atrito/resistência se desejar)
    dvdt = P * A / m_eff

    # 3. df/dt = beta * P^alpha / e_1
    # A queima para se f >= 1
    if f >= 1.0:
        dfdt = 0.0
    else:
        # Evitar P=0 dando erro em P^alpha se alpha < 1
        if P > 1e-6:  # Um limiar pequeno
            dfdt = (beta * (P ** alpha)) / e_1
        else:
            dfdt = 0.0

    return [dxdt, dvdt, dfdt]


# --- Bloco Principal de Execução ---
if __name__ == "__main__":

    # Parâmetros de exemplo (Ilustrativos, não para uma arma específica)
    # Baseado em um calibre .30-06 genérico
    params = {
        'A': np.pi * (0.00782 / 2) ** 2,  # Área (calibre 7.82mm, .308) (m^2)
        'm_proj': 0.0097,  # Massa do projétil (9.7g) (kg)
        'm_p0': 0.0032,  # Massa de propelente (3.2g) (kg)
        'rho_p': 1600.0,  # Densidade do propelente (kg/m^3)
        'V_0': 4.4e-6,  # Volume da câmara (4.4 cm^3) (m^3)
        'e_1': 0.0008,  # Espessura da alma (0.8 mm) (m)
        'beta': 5.1e-8,  # Coef. de queima (m/s / Pa^alpha)
        'alpha': 0.85,  # Expoente de queima
        'Lambda': 1.1e6,  # Força do propelente (J/kg)
        'gamma': 1.25,  # Razão de calores específicos
        'b': 1.0e-3,  # Covolume (m^3/kg)
        'theta': 0.2,  # Fator de forma (levemente progressivo)
    }

    # Condições iniciais
    y0 = [
        0.0,  # x(0) = 0
        0.0,  # v(0) = 0
        0.0001,  # f(0) = 0
    ]

    # Tempo de simulação (ex: 0 a 3 milissegundos)
    t_span = (0, 0.003)
    t_eval = np.linspace(t_span[0], t_span[1], 500)  # Pontos para salvar a solução

    # Resolver as EDOs
    sol = solve_ivp(
        internal_ballistics_model,
        t_span,
        y0,
        args=(params,),
        t_eval=t_eval,
        method='RK45'  # Method Runge-Kutta
    )

    # --- Pós-processamento para calcular a pressão ---
    # Precisamos recalcular a pressão, pois ela não é uma variável de estado
    x_sol = sol.y[0]
    v_sol = sol.y[1]
    f_sol = sol.y[2]

    # Recalcula P(t) para plotagem
    P_sol = []
    for i in range(len(sol.t)):
        x, v, f = x_sol[i], v_sol[i], f_sol[i]

        f_calc = min(f, 1.0)
        psi = (1 - params['theta']) * f_calc + params['theta'] * (f_calc ** 2)
        m_g = params['m_p0'] * psi
        V_gas = params['V_0'] + params['A'] * x - \
                (params['m_p0'] - m_g) / params['rho_p'] - m_g * params['b']
        m_eff = params['m_proj'] + params['m_p0'] / 2.0

        if V_gas <= 0:
            P = 0
        else:
            numerador = m_g * params['Lambda'] - (params['gamma'] - 1) / 2.0 * m_eff * v ** 2
            P = max(0.0, numerador / V_gas)

        P_sol.append(P)

    P_sol_MPa = np.array(P_sol) / 1e6  # Converter para Megapascals

    # --- Plotar os Resultados ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

    # 1. Pressão vs Tempo
    axs[0].plot(sol.t * 1000, P_sol_MPa, 'b-')
    axs[0].set_ylabel('Pressão (MPa)')
    axs[0].set_title('Simulação de Balística Interna (Modelo Parâmetros Concentrados)')
    axs[0].grid(True)

    # 2. Velocidade vs Tempo
    axs[1].plot(sol.t * 1000, sol.y[1], 'r-')
    axs[1].set_ylabel('Velocidade (m/s)')
    axs[1].grid(True)

    # 3. Posição vs Tempo
    axs[2].plot(sol.t * 1000, sol.y[0] * 100, 'g-')  # em cm
    axs[2].set_ylabel('Posição (cm)')
    axs[2].grid(True)

    # 4. Fração Queimada vs Tempo
    axs[3].plot(sol.t * 1000, sol.y[2], 'k-')
    axs[3].set_ylabel('Fração de Alma Queimada (f)')
    axs[3].set_xlabel('Tempo (ms)')
    axs[3].set_ylim(0, 1.1)
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    # Imprimir velocidade de saída (último valor)
    v_saida = sol.y[1][-1]
    print(f"Tempo de simulação: {t_span[1] * 1000:.1f} ms")
    print(f"Velocidade de saída (v_saida): {v_saida:.1f} m/s")
    print(f"Pressão de pico: {np.max(P_sol_MPa):.1f} MPa")
    print(f"Fração de propelente queimada: {sol.y[2][-1] * 100:.1f}%")