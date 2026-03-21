import datetime
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.io import loadmat, savemat
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt
import pyttsx3
from typing import NamedTuple
import requests


def alertar_telegram(mensagem):
    token = "8759247612:AAGX4eoLRBR8YS_VvSw8Qbc36Ivrh__Jb9I"
    chat_id = "7176139037"
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={mensagem}"
    try:
        requests.get(url)
    except Exception as e:
        print(f"Falha ao enviar notificação: {e}")


# ==========================================
# 1. FUNÇÕES AUXILIARES E CONFIGURAÇÃO
# ==========================================
def falar(texto):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        voices = engine.getProperty('voices')
        for voice in voices:
            if "brazil" in voice.name.lower() or "portuguese" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.say(texto)
        engine.runAndWait()
    except Exception as e:
        print(f"Aviso TTS: {e}")


try:
    print(f"JAX is running on: {jax.devices()[0].platform.upper()}")
except IndexError:
    print("No JAX devices found.")

# Forçar uso de float64 para evitar perda de precisão na otimização
jax.config.update("jax_enable_x64", True)

simulation_name = "Balint_SingleShooting_Scaled"
date_time_now = datetime.datetime.now()
timestamp = date_time_now.strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"results_{simulation_name}_{timestamp}.mat"

# ==========================================
# 2. CARREGAMENTO DOS DADOS DE REFERÊNCIA
# ==========================================
try:
    DATA = loadmat('Dados_Referencia.mat')
    y = DATA['P_chamber'].reshape(-1)
    time = DATA['Time'].reshape(-1)
except FileNotFoundError:
    print("\n[!] ARQUIVO 'Dados_Referencia.mat' NÃO ENCONTRADO.")
    print("Gerando dados fictícios para testar a execução do código...\n")
    time = np.linspace(0, 0.003, 500)
    y = np.sin(time * 1000) * 300e6 * np.exp(-time * 500)
    y = np.maximum(0, y)

# Adição de um ruido
noise_level = 0.02
std_deviation = noise_level * np.max(y)
noise = np.random.normal(0, std_deviation, y.shape)
y_noisy = y + noise

# Garantir que não existam pressões negativas
y = np.maximum(0, y_noisy)

decimate = 1
y = y[::decimate]
time = time[::decimate]

N = time.shape[0]
Ts = time[1] - time[0]
fs = 1 / Ts
T = time[-1]

print(f"Dataset:\nN = {N}\nfs = {fs:.2f} Hz\nT = {T:.4f} s\nTs = {Ts:.6f} s")

# Plot inicial
fig, axs = plt.subplots(1, 1, figsize=(8, 4))
axs.plot(time, y, 'k', label='Pressão Experimental (Pc)')
axs.set_title('Dados de Treinamento')
axs.set_xlabel('Tempo (s)')
axs.set_ylabel('Pressão (Pa)')
axs.legend()
axs.grid(True)
plt.tight_layout()
plt.show(block=False)

t_shot_single = jnp.array(time)
y_data_single = jnp.array(y)


# ==========================================
# 3. PARÂMETROS E MODELO FÍSICO (EDO)
# ==========================================
class FixedParams(NamedTuple):
    A: float
    m_proj: float
    m_p0: float
    rho_p: float
    V_0: float
    e_1: float
    Lambda: float
    gamma: float


static_params = FixedParams(
    A=4.8e-5, m_proj=0.0097, m_p0=0.0032, rho_p=1600.0,
    V_0=4.4e-6, e_1=0.0008, Lambda=1.1e6, gamma=1.25
)


def bal_int_jax(t, x_st, args):
    # A EDO sempre recebe e trabalha com os parâmetros FÍSICOS REAIS
    alpha, beta, b, theta, static = args
    x, v, f = x_st

    f_calc = jnp.clip(f, 0.0, 1.0)
    psi = (1 - theta) * f_calc + theta * (f_calc ** 2)
    C = static.m_p0 * psi
    M = static.m_proj

    V_gas = static.V_0 + static.A * x - (static.m_p0 - C) / static.rho_p - C * b
    m_eff_en = M + C / 3.0
    num = C * static.Lambda - (static.gamma - 1) / 2.0 * m_eff_en * v ** 2
    P_media = jnp.maximum(0.0, num / jnp.maximum(V_gas, 1e-9))

    P_base = P_media / (1 + C / (3.0 * M))
    P_camara = P_base * (1 + C / (2.0 * M))

    dxdt = v
    dvdt = (P_base * static.A) / M
    dfdt = jnp.where(f < 1.0, (beta * (P_camara ** alpha)) / static.e_1, 0.0)

    return jnp.array([dxdt, dvdt, dfdt])


term = ODETerm(bal_int_jax)
solver = Dopri5()


# ==========================================
# 4. FUNÇÃO DE CUSTO (COM ESCALONAMENTO)
# ==========================================
def create_loss_fn(t_array, y_array, static):
    @jax.jit
    def objective_jax(decision_vars_scaled):
        # 1. Recebemos as variáveis ESCALONADAS do otimizador
        alpha_s, beta_s, b_s, theta_s = decision_vars_scaled

        # 2. Desescalonamos para obter os valores físicos reais
        alpha_real = alpha_s * 1.0  # Multiplicador base 1
        beta_real = beta_s * 1e-8  # Trazemos para a escala 10^-8
        b_real = b_s * 1e-3  # Trazemos para a escala 10^-3
        theta_real = theta_s * 1.0  # Multiplicador base 1

        args = (alpha_real, beta_real, b_real, theta_real, static)

        x0 = jnp.array([0.0, 0.0, 0.0001])
        saveat = SaveAt(ts=t_array)
        sol = diffeqsolve(term, solver, t0=t_array[0], t1=t_array[-1], dt0=Ts, y0=x0, saveat=saveat, args=args)

        def model_output_step(x_step):
            x_val, v_val, f_val = x_step
            f_c = jnp.clip(f_val, 0.0, 1.0)
            psi = (1 - theta_real) * f_c + theta_real * (f_c ** 2)
            C = static.m_p0 * psi
            V_g = static.V_0 + static.A * x_val - (static.m_p0 - C) / static.rho_p - C * b_real
            m_eff = static.m_proj + C / 3.0

            num = C * static.Lambda - (static.gamma - 1) / 2.0 * m_eff * v_val ** 2
            P_m = jnp.maximum(0.0, num / jnp.maximum(V_g, 1e-9))
            P_b = P_m / (1 + C / (3.0 * static.m_proj))
            return P_b * (1 + C / (2.0 * static.m_proj))

        y_pred = jax.vmap(model_output_step)(sol.ys)

        return jnp.sum((y_pred - y_array) ** 2)

    return objective_jax


objective_jax = create_loss_fn(t_shot_single, y_data_single, static_params)
objective_grad_func = jax.jit(jax.value_and_grad(objective_jax))


def obj_for_scipy(dv_np):
    val, grad = objective_grad_func(jnp.array(dv_np))
    return np.float64(val), np.array(grad, dtype=np.float64)


# ==========================================
# 5. OTIMIZAÇÃO (L-BFGS-B COM ESCALONAMENTO)
# ==========================================
falar("Iniciando otimização rápida baseada em gradiente.")
alertar_telegram("Iniciando Otimização L-BFGS-B (Escalonada)")

# Limites ESCALONADOS
b_alpha_s = (0.5, 1.2)      # x 1.0
b_beta_s  = (0.1, 100.0)    # x 1e-8
b_b_s     = (0.1, 5.0)      # x 1e-3
b_theta_s = (0.01, 1.0)     # x 1.0
param_bounds_scaled = [b_alpha_s, b_beta_s, b_b_s, b_theta_s]

# CHUTE INICIAL ESCALONADO
# Se o físico é: alpha=0.8, beta=5e-8, b=1e-3, theta=0.15
# O escalonado fica:
initial_guess_scaled = np.array([0.8, 5.0, 1.0, 0.15])

print("\n--- Iniciando L-BFGS-B (Baseado em Gradiente + Escalonamento) ---")
with tqdm(total=500, desc="Otimizando") as pbar:
    def callback(xk):
        pbar.update(1)

    result = minimize(
        obj_for_scipy, # Como voltamos pro minimize, podemos passar a função que devolve (Valor, Gradiente)
        initial_guess_scaled,
        method='L-BFGS-B',
        jac=True, # Avisamos ao SciPy que nossa função já devolve o gradiente exato do JAX!
        bounds=param_bounds_scaled,
        options={'maxiter': 500, 'disp': False, 'ftol': 1e-9},
        callback=callback
    )

print("\nStatus da Otimização:", result.message)
# ==========================================
# 6. RESULTADOS E VALIDAÇÃO
# ==========================================
# O resultado devolve as variáveis escalonadas. Precisamos converter de volta!
alpha_opt_s, beta_opt_s, b_opt_s, theta_opt_s = result.x

alpha_opt = alpha_opt_s * 1.0
beta_opt = beta_opt_s * 1e-8
b_opt = b_opt_s * 1e-3
theta_opt = theta_opt_s * 1.0

print("\n--- Parâmetros Físicos Identificados ---")
print(f"Alpha = {alpha_opt:.6f}")
print(f"Beta  = {beta_opt:.6e}")
print(f"b     = {b_opt:.6e}")
print(f"Theta = {theta_opt:.6f}")

alertar_telegram(
    f"Otimização Concluída\nParâmetros Obtidos:\nAlpha = {alpha_opt:.6f}\nBeta  = {beta_opt:.6e}\nb     = {b_opt:.6e}\nTheta = {theta_opt:.6f}")
falar("Iniciando Simulação com dados de treinamento.")

# Rodar o modelo uma última vez com os parâmetros REAIS para gerar os gráficos
final_args = (alpha_opt, beta_opt, b_opt, theta_opt, static_params)
x0_final = jnp.array([0.0, 0.0, 0.0001])
final_sol = diffeqsolve(
    term, solver, t0=time[0], t1=time[-1], dt0=Ts,
    y0=x0_final, saveat=SaveAt(ts=t_shot_single), args=final_args
)


def model_output_step_final(x_step):
    x_val, v_val, f_val = x_step
    f_c = jnp.clip(f_val, 0.0, 1.0)
    psi = (1 - theta_opt) * f_c + theta_opt * (f_c ** 2)
    C = static_params.m_p0 * psi
    V_g = static_params.V_0 + static_params.A * x_val - (static_params.m_p0 - C) / static_params.rho_p - C * b_opt
    m_eff = static_params.m_proj + C / 3.0
    num = C * static_params.Lambda - (static_params.gamma - 1) / 2.0 * m_eff * v_val ** 2
    P_m = jnp.maximum(0.0, num / jnp.maximum(V_g, 1e-9))
    P_b = P_m / (1 + C / (3.0 * static_params.m_proj))
    return P_b * (1 + C / (2.0 * static_params.m_proj))


y_hat = jax.vmap(model_output_step_final)(final_sol.ys)

# Métricas de Erro
y_hat_np = np.array(y_hat)
MSEt = np.mean((y - y_hat_np) ** 2)
y_mean = np.mean(y)
RSS = np.sum((y - y_hat_np) ** 2)
TSS = np.sum((y - y_mean) ** 2)
r2t = 1.0 - (RSS / TSS)

alertar_telegram(f"Métricas de Ajuste:\nR² = {r2t:.6f}\nMSE = {MSEt:.6e}")
print(f"\nMétricas:\nR² = {r2t:.6f}\nMSE = {MSEt:.6e}")

# Gráfico Final
plt.figure(figsize=(12, 6))
plt.plot(time, y, 'k', label='Dados Experimentais (Com Ruído)', alpha=0.6)
plt.plot(time, y_hat_np, 'b--', label='Modelo Identificado', linewidth=2)
plt.plot(time, y - y_hat_np, 'r', label='Resíduo (Erro)', linewidth=1.5, alpha=0.7)
plt.xlabel('Tempo (s)')
plt.ylabel('Pressão na Câmara (Pa)')
plt.title('Identificação do Modelo de Balística Interna (Evolução Diferencial Escalonada)')
plt.legend()
plt.grid(True)

# Salvando Resultados
sim_results = {
    'Params_Estimados': np.array([alpha_opt, beta_opt, b_opt, theta_opt]),
    'y_experimental': y,
    'y_hat_simulado': y_hat_np,
    'Tempo': time,
    'Train_Metrics_R2_MSE': np.array([r2t, MSEt])
}

try:
    savemat(file_name, sim_results, do_compression=True)
    print(f"\nResultados salvos com sucesso em: {file_name}")
except Exception as e:
    print(f"\nErro ao salvar os resultados: {e}")

falar("Simulação encerrada.")
plt.show()