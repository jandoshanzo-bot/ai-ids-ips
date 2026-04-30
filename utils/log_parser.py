"""
Log Parser for IDS — Nginx/Apache лог файлдарын талдайды
=====================================================================
Не істейді:
  1. access.log файлын үздіксіз оқиды (tail -f сияқты)
  2. Әр жолды талдап, желі белгілерін (features) шығарады
  3. IDS моделіне береді — шабуыл ба, қалыпты ма?
  4. Шабуыл болса — дерекқорға сақтайды, Telegram-ға жібереді
  5. Flask SSE (Server-Sent Events) арқылы браузерге нақты уақытта хабарлайды

Қолданылған технологиялар:
  - re (Regular Expressions)   — лог жолдарын бөлшектеу
  - threading                  — фондық процесс (негізгі Flask-пен қатар жұмыс)
  - time                       — жаңа жол күту циклі
  - collections.defaultdict    — IP статистикасын жинау
  - datetime                   — уақыт есептеу
"""

from __future__ import annotations

import re
import time
import threading
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)

# ── Nginx combined log форматы ─────────────────────────────────────────────
# Мысал: 192.168.1.1 - - [05/Mar/2026:10:30:00 +0000] "GET /index.html HTTP/1.1" 200 1234 "-" "Mozilla/5.0"
NGINX_PATTERN = re.compile(
    r'(?P<ip>\S+)'           # IP мекенжайы
    r'\s+\S+\s+\S+'          # ident, auth (әдетте "-")
    r'\s+\[(?P<time>[^\]]+)\]'  # уақыт
    r'\s+"(?P<method>\S+)'   # HTTP методы
    r'\s+(?P<path>\S+)'      # URL жолы
    r'\s+(?P<proto>[^"]+)"'  # протокол
    r'\s+(?P<status>\d{3})'  # статус коды
    r'\s+(?P<bytes>\d+|-)'   # жауап өлшемі (байт)
    r'(?:\s+"(?P<referer>[^"]*)")?'   # referer (міндетті емес)
    r'(?:\s+"(?P<agent>[^"]*)")?'     # user-agent (міндетті емес)
)

# Apache combined log — nginx-пен бірдей формат
APACHE_PATTERN = NGINX_PATTERN

# ── Белгілі шабуыл паттерндері ─────────────────────────────────────────────
# Бұл ережелер `flag` және басқа белгілерді бастапқы мәнге қою үшін қолданылады
ATTACK_URL_PATTERNS = [
    (re.compile(r'(\.\.\/|%2e%2e)', re.I),       'path_traversal'),   # Directory traversal
    (re.compile(r'(<script|javascript:|onerror)', re.I), 'xss'),       # XSS
    (re.compile(r'(union\s+select|drop\s+table|insert\s+into)', re.I), 'sqli'),  # SQL Injection
    (re.compile(r'(wp-login|phpmyadmin|\.env|\.git)', re.I), 'scan'),  # Сканерлеу
    (re.compile(r'(cmd=|exec=|system\(|passthru)', re.I), 'rce'),      # RCE
    (re.compile(r'(/etc/passwd|/etc/shadow)', re.I), 'lfi'),           # LFI
]

# ── Сервис анықтау (URL → IDS сервис белгісі) ─────────────────────────────
SERVICE_MAP = {
    '/':        'http',
    'login':    'http',
    'admin':    'http',
    'api':      'http',
    'ftp':      'ftp',
    'smtp':     'smtp',
    'mail':     'smtp',
    'ssh':      'ssh',
    'database': 'private',
    'db':       'private',
    'wp-':      'http',
    'php':      'http',
}


def parse_log_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Лог жолын талдайды және IDS белгілерін шығарады.

    Nginx/Apache combined format жолынан:
      → IP, уақыт, метод, URL, статус, байт саны алынады
      → IDS моделіне қажетті 41 белгіге айналдырылады

    Args:
        line: Nginx/Apache access.log-тің бір жолы

    Returns:
        IDS белгілері сөздігі немесе None (жол дұрыс форматта болмаса)
    """
    line = line.strip()
    if not line:
        return None

    m = NGINX_PATTERN.match(line)
    if not m:
        return None

    ip        = m.group('ip')
    method    = m.group('method').upper()
    path      = m.group('path')
    proto     = m.group('proto').strip()   # 'HTTP/1.1'
    status    = int(m.group('status'))
    raw_bytes = m.group('bytes')
    agent     = m.group('agent') or ''

    # Байт санын сандық мәнге айналдыру
    dst_bytes = int(raw_bytes) if raw_bytes and raw_bytes != '-' else 0

    # ── Протокол белгісін анықтау ─────────────────────────────────────────
    if 'HTTPS' in proto or ':443' in path:
        protocol_type = 'tcp'
    elif method in ('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'):
        protocol_type = 'tcp'
    else:
        protocol_type = 'udp'

    # ── Сервис белгісі ────────────────────────────────────────────────────
    service = 'http'
    path_lower = path.lower()
    for keyword, svc in SERVICE_MAP.items():
        if keyword in path_lower:
            service = svc
            break

    # ── Flag белгісі (TCP қосылым күйі) ──────────────────────────────────
    # HTTP статус кодтарын TCP flag-тарына сәйкестендіру:
    #   200-299 → SF  (Successfully Finished — сәтті аяқталған)
    #   400-499 → REJ (Rejected — қабылданбады)
    #   500-599 → RSTO (Reset by server — сервер тарапынан үзілді)
    #   301-302 → S1  (SYN қабылданды)
    if 200 <= status < 300:
        flag = 'SF'
    elif 400 <= status < 500:
        flag = 'REJ'
    elif 500 <= status < 600:
        flag = 'RSTO'
    elif 300 <= status < 400:
        flag = 'S1'
    else:
        flag = 'SF'

    # ── Шабуыл паттерндерін тексеру ──────────────────────────────────────
    attack_hint = None
    for pattern, name in ATTACK_URL_PATTERNS:
        if pattern.search(path) or pattern.search(agent):
            attack_hint = name
            break

    # ── Сандық белгілер ───────────────────────────────────────────────────
    # src_bytes: HTTP сұраныс өлшемін болжалды есептеу
    # (нақты мән тек пакет деңгейінде белгілі, лог-та жоқ)
    method_size = {'GET': 200, 'POST': 1500, 'PUT': 2000, 'DELETE': 150}.get(method, 300)
    src_bytes = method_size + len(path) * 2

    # Шабуыл белгілері болса — src_bytes жоғарылатамыз
    if attack_hint:
        src_bytes *= 3
        flag = 'S0' if attack_hint in ('sqli', 'rce', 'lfi') else 'REJ'

    # ── IDS белгілер сөздігі ─────────────────────────────────────────────
    features = {
        # Негізгі желі белгілері
        'duration':           0,                     # HTTP stateless — ұзақтық жоқ
        'protocol_type':      protocol_type,         # tcp / udp
        'service':            service,               # http / ftp / smtp
        'flag':               flag,                  # SF / REJ / S0 / RSTO
        'src_bytes':          src_bytes,             # Клиент жіберген байт
        'dst_bytes':          dst_bytes,             # Сервер жауабы (байт)
        'land':               0,
        'wrong_fragment':     0,
        'urgent':             0,

        # Кіру белгілері
        'hot':                1 if attack_hint else 0,
        'num_failed_logins':  1 if status == 401 else 0,
        'logged_in':          1 if status in (200, 201, 204) else 0,
        'num_compromised':    1 if attack_hint in ('rce', 'lfi') else 0,
        'root_shell':         1 if attack_hint == 'rce' else 0,
        'su_attempted':       0,
        'num_root':           1 if attack_hint == 'rce' else 0,
        'num_file_creations': 1 if method in ('PUT', 'POST') and status == 201 else 0,
        'num_shells':         0,
        'num_access_files':   1 if attack_hint == 'lfi' else 0,
        'num_outbound_cmds':  0,
        'is_host_login':      0,
        'is_guest_login':     1 if status == 401 else 0,

        # Қосылым статистикасы (IP бойынша санақ — төменде толтырылады)
        'count':              1,
        'srv_count':          1,
        'serror_rate':        1.0 if flag in ('S0', 'REJ') else 0.0,
        'srv_serror_rate':    1.0 if flag in ('S0', 'REJ') else 0.0,
        'rerror_rate':        1.0 if flag == 'RSTO' else 0.0,
        'srv_rerror_rate':    1.0 if flag == 'RSTO' else 0.0,
        'same_srv_rate':      1.0,
        'diff_srv_rate':      0.0,
        'srv_diff_host_rate': 0.0,

        # Host-деңгейіндегі статистика
        'dst_host_count':          100,
        'dst_host_srv_count':      100,
        'dst_host_same_srv_rate':  1.0,
        'dst_host_diff_srv_rate':  0.1 if attack_hint == 'scan' else 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate':    1.0 if flag in ('S0', 'REJ') else 0.0,
        'dst_host_srv_serror_rate':1.0 if flag in ('S0', 'REJ') else 0.0,
        'dst_host_rerror_rate':    1.0 if flag == 'RSTO' else 0.0,
        'dst_host_srv_rerror_rate':1.0 if flag == 'RSTO' else 0.0,

        # Мета-деректер (IDS белгісі емес, хабарлама үшін)
        '_ip':           ip,
        '_method':       method,
        '_path':         path,
        '_status':       status,
        '_attack_hint':  attack_hint,
        '_raw_line':     line,
    }

    return features


class IPTracker:
    """
    Бір IP мекенжайынан келетін сұраныстарды бақылайды.

    Не істейді:
      - Соңғы 2 минуттағы сұраныстарды санайды
      - Серия шабуылдарын (DoS, brute force) анықтайды
      - IDS-тің `count`, `srv_count`, `serror_rate` белгілерін нақтылайды

    Қолданылған: defaultdict — Python стандартты кітапханасы
    """

    def __init__(self, window_seconds: int = 120):
        self.window = window_seconds
        # ip → [(timestamp, service, is_error), ...]
        self._history: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, ip: str, service: str, is_error: bool) -> Dict[str, float]:
        """IP-дың соңғы window_seconds ішіндегі статистикасын қайтарады."""
        now = time.time()
        cutoff = now - self.window

        with self._lock:
            # Ескі жазбаларды тазалау
            self._history[ip] = [
                r for r in self._history[ip] if r[0] > cutoff
            ]
            # Жаңа жазба қосу
            self._history[ip].append((now, service, is_error))
            records = self._history[ip]

        total   = len(records)
        errors  = sum(1 for r in records if r[2])
        srv_cnt = len(set(r[1] for r in records))

        return {
            'count':       min(total, 511),         # IDS максималды мәні
            'srv_count':   min(srv_cnt, 511),
            'serror_rate': errors / total if total else 0.0,
            'diff_srv_rate': (srv_cnt - 1) / total if total > 1 else 0.0,
        }

    def is_suspicious(self, ip: str, threshold: int = 100) -> bool:
        """Соңғы 2 минутта threshold-тан артық сұраныс болса — күдікті."""
        with self._lock:
            return len(self._history.get(ip, [])) > threshold


class LogWatcher:
    """
    Лог файлын үздіксіз оқитын негізгі класс.

    Жұмыс принципі (tail -f):
      1. Файлды ашады, соңына жылжиды
      2. Жаңа жол келгенде — parse_log_line() шақырады
      3. IPTracker арқылы статистиканы толтырады
      4. IDS моделіне береді
      5. Нәтижені callback арқылы Flask-ке жібереді

    Қолданылған:
      - threading.Thread  — фондық ағын
      - threading.Event   — тоқтату сигналы
      - file.seek()       — файл соңына өту
      - file.tell()       — позицияны есте сақтау (файл ротациясын анықтау)
    """

    def __init__(
        self,
        log_path: str,
        on_result: Callable[[Dict], None],
        poll_interval: float = 0.5,
    ):
        """
        Args:
            log_path:      Nginx/Apache access.log файлының жолы
            on_result:     Нәтиже дайын болғанда шақырылатын функция
            poll_interval: Жаңа жол іздеу жиілігі (секунд)
        """
        self.log_path      = Path(log_path)
        self.on_result     = on_result
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ip_tracker = IPTracker()

        # Статистика
        self.total_lines    = 0
        self.total_attacks  = 0
        self.is_running     = False

    def start(self) -> bool:
        """Фондық ағынды іске қосу."""
        if not self.log_path.exists():
            logger.error("Лог файлы табылмады: %s", self.log_path)
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name='LogWatcher',
            daemon=True,   # Flask тоқтағанда автоматты тоқтайды
        )
        self._thread.start()
        self.is_running = True
        logger.info("LogWatcher іске қосылды: %s", self.log_path)
        return True

    def stop(self):
        """Фондық ағынды тоқтату."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.is_running = False
        logger.info("LogWatcher тоқтатылды.")

    def _watch_loop(self):
        """
        Негізгі цикл — файлды үздіксіз оқиды.
        tail -f командасының Python баламасы.
        """
        with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
            # Файл соңына өту — тек жаңа жолдарды оқу үшін
            f.seek(0, 2)
            last_pos  = f.tell()
            last_size = self.log_path.stat().st_size

            while not self._stop_event.is_set():
                # Файл ротациясын тексеру (logrotate жағдайы)
                try:
                    current_size = self.log_path.stat().st_size
                except FileNotFoundError:
                    time.sleep(1)
                    continue

                if current_size < last_size:
                    # Файл жаңартылды — басынан оқу
                    f.seek(0)
                    logger.info("Лог файлы ротацияланды, басынан оқылады.")

                last_size = current_size
                line = f.readline()

                if not line:
                    # Жаңа жол жоқ — poll_interval күту
                    time.sleep(self.poll_interval)
                    continue

                last_pos = f.tell()
                self._process_line(line)

    def _process_line(self, line: str):
        """
        Бір лог жолын өңдейді:
          1. parse_log_line() → белгілер
          2. IPTracker → count, serror_rate жаңарту
          3. IDS моделі → болжам
          4. on_result callback → Flask-ке жіберу
        """
        features = parse_log_line(line)
        if not features:
            return

        self.total_lines += 1
        ip      = features.get('_ip', '')
        service = features.get('service', 'http')
        is_err  = features.get('serror_rate', 0) > 0

        # IP статистикасын жаңарту
        ip_stats = self._ip_tracker.record(ip, service, is_err)
        features.update(ip_stats)

        # Мета-деректерді бөлу (моделге берілмейді)
        meta = {k: features.pop(k) for k in list(features) if k.startswith('_')}

        # on_result арқылы Flask route-ке жіберу
        # (нақты болжам app.py ішінде жасалады)
        result = {
            'features': features,
            'meta':     meta,
            'ip':       ip,
            'timestamp': datetime.now().isoformat(),
            'is_suspicious': self._ip_tracker.is_suspicious(ip),
        }
        self.on_result(result)


# ── Глобалды LogWatcher нысаны ─────────────────────────────────────────────
_watcher: Optional[LogWatcher] = None


def get_watcher() -> Optional[LogWatcher]:
    return _watcher


def start_watcher(log_path: str, on_result: Callable) -> bool:
    """LogWatcher-ді іске қосу (app.py-дан шақырылады)."""
    global _watcher
    if _watcher and _watcher.is_running:
        _watcher.stop()
    _watcher = LogWatcher(log_path, on_result)
    return _watcher.start()


def stop_watcher():
    """LogWatcher-ді тоқтату."""
    global _watcher
    if _watcher:
        _watcher.stop()
        _watcher = None
