# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:32:08 2026

@author: IDRZBOOK002
"""

# python
import sys
import subprocess
from pathlib import Path
import requests


def alertar_telegram(mensagem):
    token = "8759247612:AAGX4eoLRBR8YS_VvSw8Qbc36Ivrh__Jb9I"
    chat_id = "7176139037"
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={mensagem}"
    try:
        requests.get(url)
    except Exception as e:
        print(f"Falha ao enviar notificação: {e}")


def run_scripts_in_sequence(scripts, continue_on_error=False):
    """
    Executa uma lista de scripts Python em sequência usando o mesmo
    interpretador (sys.executable). Retorna True se todos terminaram OK.
    """
    py = sys.executable  # garante usar o mesmo Python do PyCharm
    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"Arquivo não encontrado: `{script_path}`")
            if not continue_on_error:
                return False
            else:
                continue
        print(f"Executando `{script_path}` ...")
        alertar_telegram(f"Inciando execução do código: {script}")
        try:
            res = subprocess.run([py, str(script_path)], check=True)
            print(f"`{script_path}` finalizado com código {res.returncode}\n")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar `{script_path}`: código {e.returncode}")
            if not continue_on_error:
                return False
    return True

if __name__ == "__main__":
    # Exemplo: liste aqui os seus scripts na ordem desejada
    scripts = [
        #"ID_Grey_Balint_7.py",
        #"ID_Grey_Balint_6.py",
        #"ID_Grey_Balint_5.py",
        #"ID_Grey_Balint_4.py",
        "ID_Grey_Balint_3.py",
        "ID_Grey_Balint_2.py",
        "ID_Grey_Balint_1.py",
        "ID_Grey_Balint_0.py",
        ]
    alertar_telegram("Iniciando sequência de execução dos códigos baseados em evolução diferencial e com ruído ...")
    ok = run_scripts_in_sequence(scripts, continue_on_error=False)
    print("Sequência concluída com sucesso." if ok else "Sequência interrompida por erro.")