#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ Colibrì GLM-5.2 — Прямой интерактивный консольный чат (Zero-Context Bloat)
"""

import os
import sys
import json
import time
import requests

# Настройка кодировки для Windows консоли
if sys.platform == "win32":
    os.system("chcp 65001 >nul 2>&1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

API_URL = os.environ.get("COLI_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL_ID = os.environ.get("COLI_MODEL_ID", "calibry-glm-5.2")

# ANSI цвета
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

BANNER = f"""
{CYAN}{BOLD}╔════════════════════════════════════════════════════════════════════════╗
║   🚀 COLIBRÌ GLM-5.2 (744B MoE) — ПРЯМОЙ КОНСОЛЬНЫЙ ТЕРМИНАЛ           ║
║   ⚡ GPU: RTX 3090 (GDDR6X @ 936 GB/s) · MTP Speculation: Active        ║
╚════════════════════════════════════════════════════════════════════════╝{RESET}
{GRAY}Команды: {YELLOW}/reset{GRAY} — очистить историю, {YELLOW}/exit{GRAY} — выход{RESET}
"""

def main():
    print(BANNER)
    history = []
    
    # Проверка доступности сервера
    try:
        r = requests.get("http://127.0.0.1:8000/v1/models", timeout=3)
        if r.status_code == 200:
            print(f"{GREEN}✓ Сервер Colibrì активен и готов к генерации!{RESET}\n")
    except Exception:
        print(f"{YELLOW}⚠ Ожидание готовности сервера http://127.0.0.1:8000/v1 ...{RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}Вы › {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GRAY}Сессия завершена.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "exit", "quit", ":q"):
            print(f"{GRAY}До свидания!{RESET}")
            break

        if user_input.lower() in ("/reset", "reset", ":reset", "/clear"):
            history.clear()
            print(f"\n{YELLOW}↺ История диалога очищена. Начат новый диалог без контекста.{RESET}\n")
            continue

        history.append({"role": "user", "content": user_input})

        payload = {
            "model": MODEL_ID,
            "messages": history,
            "stream": True,
            "temperature": 0.7,
            "top_p": 0.90,
            "max_tokens": 512,
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>", "<|system|>"]
        }

        print(f"\n{CYAN}{BOLD}GLM-5.2 › {RESET}", end="", flush=True)

        full_response = ""
        tokens_count = 0
        t_start = time.time()
        t_first_token = None

        try:
            resp = requests.post(API_URL, json=payload, stream=True, timeout=300)
            if resp.status_code != 200:
                print(f"\n{YELLOW}[Ошибка сервера {resp.status_code}: {resp.text}]{RESET}")
                continue

            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="replace")
                if decoded.startswith("data: "):
                    data_str = decoded[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            if t_first_token is None:
                                t_first_token = time.time()
                            tokens_count += 1
                            full_response += delta
                            sys.stdout.write(f"{WHITE}{delta}{RESET}")
                            sys.stdout.flush()
                    except Exception:
                        pass

            t_end = time.time()
            dt_total = t_end - t_start
            dt_prefill = (t_first_token - t_start) if t_first_token else dt_total
            dt_gen = (t_end - t_first_token) if t_first_token else 0.001
            tps = tokens_count / dt_gen if dt_gen > 0 else 0

            print(f"\n\n{GRAY}[⏱️ Префилл: {dt_prefill:.1f}с · Генерация: {dt_gen:.1f}с · {tokens_count} токенов ({tps:.2f} tok/s)]{RESET}\n")
            history.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            print(f"\n{YELLOW}[Ошибка подключения к серверу: {e}]{RESET}\n")

if __name__ == "__main__":
    main()
