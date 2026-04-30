"""
simulate_log.py — Nginx лог симуляторы (демонстрация үшін)
===========================================================
Не істейді:
  Нақты Nginx/Apache лог форматында жалған жолдар жазады.
  Бұл скрипт арқылы нақты сервер болмаса да тікелей
  мониторингті мұғалімге немесе комиссияға көрсетуге болады.

Іске қосу:
  python simulate_log.py --log /tmp/test_access.log

Содан кейін .env файлына:
  LOG_PATH=/tmp/test_access.log

Қолданылған:
  - argparse   — командалық жол аргументтері
  - random     — кездейсоқ деректер генерациясы
  - time       — жазу аралықтарын баяулату
  - datetime   — Nginx уақыт форматы
"""

import argparse
import random
import time
import os
from pathlib import Path
from datetime import datetime, timezone

# Үнсіз мән — simulate_log.py-дің өзі тұрған папка
DEFAULT_LOG = str(Path(__file__).parent / 'test_access.log')

# ── Шаблон деректер ────────────────────────────────────────
NORMAL_IPS = [
    '192.168.1.10', '192.168.1.15', '10.0.0.5',
    '172.16.0.100', '192.168.2.20', '10.10.5.3',
]

ATTACK_IPS = [
    '185.220.101.5',   # Tor exit node
    '45.33.32.156',    # Сканерлеуші
    '198.51.100.42',   # Болжамды шабуылшы
    '203.0.113.99',    # Тест IP
]

NORMAL_PATHS = [
    ('GET',  '/', 200, 1234),
    ('GET',  '/index.html', 200, 5678),
    ('GET',  '/about', 200, 890),
    ('POST', '/api/login', 200, 234),
    ('GET',  '/static/main.css', 200, 4321),
    ('GET',  '/favicon.ico', 200, 1024),
    ('GET',  '/api/health', 200, 89),
    ('GET',  '/products', 200, 7654),
]

ATTACK_PATHS = [
    ('GET',  '/../../../etc/passwd', 400, 571),          # Path traversal
    ('POST', '/wp-login.php', 403, 234),                 # WordPress scan
    ('GET',  '/phpmyadmin/', 404, 189),                  # phpMyAdmin scan
    ('GET',  '/?id=1 UNION SELECT', 400, 450),           # SQL Injection
    ('GET',  '/<script>alert(1)</script>', 400, 312),    # XSS
    ('GET',  '/.env', 403, 145),                         # Env file leak
    ('GET',  '/.git/config', 403, 234),                  # Git leak
    ('POST', '/admin/exec?cmd=whoami', 403, 89),         # RCE
    ('GET',  '/etc/shadow', 403, 67),                    # LFI
    ('GET',  '/login?user=admin&pass=admin', 401, 445),  # Brute force
]

AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    'Mozilla/5.0 (Linux; Android 10)',
    'curl/7.68.0',
    'python-requests/2.28.0',
    'Nikto/2.1.6',           # Сканер
    'sqlmap/1.7',            # SQL injection tool
    'masscan/1.3',           # Port scanner
]


def nginx_time(dt: datetime) -> str:
    """Nginx уақыт форматы: 05/Mar/2026:10:30:00 +0000"""
    return dt.strftime('%d/%b/%Y:%H:%M:%S +0000')


def make_log_line(ip: str, method: str, path: str, status: int,
                  size: int, agent: str) -> str:
    """Nginx combined log форматындағы бір жол жасау."""
    now = datetime.now(timezone.utc)
    return (
        f'{ip} - - [{nginx_time(now)}] '
        f'"{method} {path} HTTP/1.1" '
        f'{status} {size} "-" "{agent}"'
    )


def main():
    parser = argparse.ArgumentParser(description='IDS Nginx лог симуляторы')
    parser.add_argument('--log',      default=DEFAULT_LOG,
                        help='Лог файлы жолы (үнсіз: ids_project/test_access.log)')
    parser.add_argument('--rate',     type=float, default=0.5,
                        help='Жол жазу аралығы (секунд)')
    parser.add_argument('--attack-ratio', type=float, default=0.25,
                        help='Шабуыл үлесі 0-1 (үнсіз: 0.25)')
    args = parser.parse_args()

    log_file = Path(args.log).resolve()
    print(f"📝 Лог файлы: {log_file}")
    print(f"⏱  Аралық: {args.rate} сек")
    print(f"⚠️  Шабуыл үлесі: {args.attack_ratio * 100:.0f}%")
    print()
    print("=" * 60)
    print(f"  Браузерде осы жолды енгізіңіз:")
    print(f"  {log_file}")
    print("=" * 60)
    print("Тоқтату үшін: Ctrl+C\n")

    with open(log_file, 'a', encoding='utf-8') as f:
        count = 0
        while True:
            is_attack = random.random() < args.attack_ratio

            if is_attack:
                ip             = random.choice(ATTACK_IPS)
                method, path, status, size = random.choice(ATTACK_PATHS)
                agent          = random.choice(AGENTS[-3:])   # Шабуыл агенттері
                # DoS симуляциясы: кейде бір IP-дан бірнеше рет жібер
                burst = random.randint(1, 5) if random.random() < 0.2 else 1
            else:
                ip             = random.choice(NORMAL_IPS)
                method, path, status, size = random.choice(NORMAL_PATHS)
                agent          = random.choice(AGENTS[:3])
                burst          = 1

            for _ in range(burst):
                line = make_log_line(ip, method, path, status, size, agent)
                f.write(line + '\n')
                f.flush()   # Дереу дискіге жазу — LogWatcher оқу үшін
                count += 1

                label = '🔴 ШАБУЫЛ' if is_attack else '🟢 Normal'
                print(f"[{count:4d}] {label}  {ip:<18} {method} {path[:50]}")

            time.sleep(args.rate)


if __name__ == '__main__':
    main()