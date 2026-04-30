"""
Attack Type Classifier for IDS
Шабуыл типін анықтау модулі — KDD Cup белгілері негізінде

Шабуыл категориялары:
  DoS   — қызметтен бас тарту шабуылы
  Probe — желіні барлау/сканерлеу
  R2L   — қашықтан жергілікті кіру
  U2R   — пайдаланушыдан root-ке өту
"""

from __future__ import annotations
from typing import Dict, Any


# ── Шабуыл типтерінің сипаттамасы ───────────────────────────────────────────
ATTACK_META: Dict[str, Dict[str, str]] = {
    "DDoS": {
        "full_name": "Distributed Denial of Service",
        "kk": "Таратылған қызметтен бас тарту",
        "emoji": "⚡",
        "color": "#ef4444",
        "badge": "danger",
        "description": "Many sources overwhelm a target service with traffic.",
        "severity": "Critical",
    },
    "PortScan": {
        "full_name": "Port scanning",
        "kk": "Порттарды сканерлеу",
        "emoji": "🔎",
        "color": "#f59e0b",
        "badge": "warning",
        "description": "Reconnaissance activity that probes open ports and services.",
        "severity": "Medium",
    },
    "Bot": {
        "full_name": "Botnet traffic",
        "kk": "Ботнет трафигі",
        "emoji": "🤖",
        "color": "#06b6d4",
        "badge": "info",
        "description": "Automated malicious traffic from compromised hosts.",
        "severity": "High",
    },
    "DoS Hulk": {
        "full_name": "DoS Hulk",
        "kk": "DoS Hulk шабуылы",
        "emoji": "💥",
        "color": "#dc2626",
        "badge": "danger",
        "description": "High-volume HTTP denial-of-service traffic.",
        "severity": "High",
    },
    "DoS GoldenEye": {
        "full_name": "DoS GoldenEye",
        "kk": "DoS GoldenEye шабуылы",
        "emoji": "💥",
        "color": "#b91c1c",
        "badge": "danger",
        "description": "HTTP denial-of-service traffic with randomized requests.",
        "severity": "High",
    },
    "DoS slowloris": {
        "full_name": "DoS slowloris",
        "kk": "DoS slowloris шабуылы",
        "emoji": "⏳",
        "color": "#f97316",
        "badge": "warning",
        "description": "Slow HTTP connection exhaustion attack.",
        "severity": "High",
    },
    "DoS Slowhttptest": {
        "full_name": "DoS Slowhttptest",
        "kk": "DoS Slowhttptest шабуылы",
        "emoji": "⏳",
        "color": "#ea580c",
        "badge": "warning",
        "description": "Slow HTTP request/response denial-of-service pattern.",
        "severity": "High",
    },
    "FTP-Patator": {
        "full_name": "FTP brute force",
        "kk": "FTP brute force",
        "emoji": "🔐",
        "color": "#8b5cf6",
        "badge": "purple",
        "description": "Repeated FTP login attempts.",
        "severity": "High",
    },
    "SSH-Patator": {
        "full_name": "SSH brute force",
        "kk": "SSH brute force",
        "emoji": "🔐",
        "color": "#7c3aed",
        "badge": "purple",
        "description": "Repeated SSH login attempts.",
        "severity": "High",
    },
    "Web Attack Brute Force": {
        "full_name": "Web attack brute force",
        "kk": "Web brute force шабуылы",
        "emoji": "🌐",
        "color": "#ec4899",
        "badge": "pink",
        "description": "Repeated authentication attempts against a web application.",
        "severity": "High",
    },
    "Web Attack XSS": {
        "full_name": "Cross-site scripting",
        "kk": "XSS веб-шабуылы",
        "emoji": "🌐",
        "color": "#db2777",
        "badge": "pink",
        "description": "Malicious script injection attempt against a web application.",
        "severity": "Medium",
    },
    "Web Attack SQL Injection": {
        "full_name": "SQL injection",
        "kk": "SQL Injection веб-шабуылы",
        "emoji": "🧩",
        "color": "#be123c",
        "badge": "danger",
        "description": "Attempt to manipulate database queries through user input.",
        "severity": "Critical",
    },
    "Infiltration": {
        "full_name": "Infiltration",
        "kk": "Жүйеге жасырын ену",
        "emoji": "🕵",
        "color": "#64748b",
        "badge": "secondary",
        "description": "Suspicious traffic associated with unauthorized internal access.",
        "severity": "Critical",
    },
    "Heartbleed": {
        "full_name": "Heartbleed",
        "kk": "Heartbleed осалдығы",
        "emoji": "🩸",
        "color": "#991b1b",
        "badge": "danger",
        "description": "Traffic associated with the OpenSSL Heartbleed vulnerability.",
        "severity": "Critical",
    },
    "DoS": {
        "full_name": "Қызмет көрсетуден бас тарту шабуылы",
        "kk": "Қызмет көрсетуден бас тарту шабуылы",
        "emoji": "💥",
        "color": "#ef4444",
        "badge": "danger",
        "description": "Сервисті шамадан тыс сұраныспен толтырып, қолжетімсіз ету",
        "severity": "Жоғары",
    },
    "Probe": {
        "full_name": "Барлау шабуылы",
        "kk": "Барлау шабуылы",
        "emoji": "🔍",
        "color": "#f59e0b",
        "badge": "warning",
        "description": "Желіні, порттарды немесе осалдықтарды іздестіру",
        "severity": "Орташа",
    },
    "R2L": {
        "full_name": "Қашықтан жүйеге кіру шабуылы",
        "kk": "Қашықтан жүйеге кіру шабуылы",
        "emoji": "🔓",
        "color": "#8b5cf6",
        "badge": "purple",
        "description": "Рұқсатсыз жергілікті есептік жазбаға қол жеткізу",
        "severity": "Жоғары",
    },
    "U2R": {
        "full_name": "Пайдаланушыдан әкімшіге өту шабуылы",
        "kk": "Пайдаланушыдан әкімшіге өту шабуылы",
        "emoji": "👑",
        "color": "#ec4899",
        "badge": "pink",
        "description": "Қарапайым пайдаланушы аккаунтынан root-ке ену",
        "severity": "Өте жоғары",
    },
    "Normal": {
        "full_name": "Қалыпты трафик",
        "kk": "Қалыпты",
        "emoji": "✅",
        "color": "#10b981",
        "badge": "success",
        "description": "Шабуыл анықталмады",
        "severity": "Жоқ",
    },
    "Unknown": {
        "full_name": "Белгісіз шабуыл",
        "kk": "Белгісіз шабуыл",
        "emoji": "❓",
        "color": "#6b7280",
        "badge": "secondary",
        "description": "Белгісіз шабуыл үлгісі",
        "severity": "Белгісіз",
    },
}


def classify_attack_type(features: Dict[str, Any], probability: float) -> Dict[str, Any]:
    """
    Берілген желі трафик белгілері мен болжам ықтималдығы негізінде
    шабуыл типін анықтайды.

    KDD Cup 1999 белгілерінің мәніне қарай эвристикалық ережелер:

    DoS  — serror_rate жоғары, count жоғары, src_bytes жоғары, flag=S0/REJ
    Probe — dst_host_diff_srv_rate жоғары, srv_count төмен, duration қысқа
    R2L  — num_failed_logins > 0, logged_in=0, duration ұзын, is_guest_login
    U2R  — root_shell=1, su_attempted=1, num_root > 0, num_file_creations > 0

    Args:
        features: predict() нәтижесіндегі белгілер сөздігі
        probability: шабуыл ықтималдығы (0-1)

    Returns:
        {
          "type": "DoS" | "Probe" | "R2L" | "U2R" | "Normal" | "Unknown",
          "confidence": float,   # шабуыл типіне сенімділік (0-1)
          "scores": {...},       # әр типтің ұпайы
          "meta": {...}          # ATTACK_META ішіндегі сипаттама
        }
    """
    if probability < 0.3:
        return _build_result("Normal", 1.0 - probability, {})

    f = _safe_features(features)

    scores: Dict[str, float] = {
        "DoS":   _score_dos(f),
        "Probe": _score_probe(f),
        "R2L":   _score_r2l(f),
        "U2R":   _score_u2r(f),
    }

    # Ықтималдыққа пропорционалды масштаблау
    total = sum(scores.values()) or 1.0
    normed = {k: v / total for k, v in scores.items()}

    best_type = max(normed, key=normed.__getitem__)
    best_conf = normed[best_type] * probability  # шабуыл ықтималдығымен кескіндеу

    # Егер барлық ұпайлар тым төмен болса — Unknown
    if normed[best_type] < 0.30:
        best_type = "Unknown"

    return _build_result(best_type, round(best_conf, 4), normed)


# ── Ішкі ұпай есептеу функциялары ───────────────────────────────────────────

def _score_dos(f: Dict[str, float]) -> float:
    score = 0.0
    # Жоғары қосылым саны → DoS
    if f["count"] > 100:
        score += 0.40
    elif f["count"] > 50:
        score += 0.20
    # Жоғары serror_rate → SYN flood (ең күшті DoS белгісі)
    if f["serror_rate"] > 0.7:
        score += 0.40
    elif f["serror_rate"] > 0.4:
        score += 0.20
    # flag S0 немесе REJ → аяқталмаған қосылымдар
    if f["flag_s0"] or f["flag_rej"]:
        score += 0.20
    # Өте жоғары src_bytes → bandwidth flood
    if f["src_bytes"] > 10000:
        score += 0.15
    # ICMP болса DoS емес
    if f["proto_icmp"]:
        score -= 0.30
    return max(0.0, min(score, 1.0))


def _score_probe(f: Dict[str, float]) -> float:
    score = 0.0
    # Әр түрлі сервистерге сұраныс → порт сканерлеу (ең күшті белгі)
    if f["dst_host_diff_srv_rate"] > 0.6:
        score += 0.40
    elif f["dst_host_diff_srv_rate"] > 0.3:
        score += 0.25
    # diff_srv_rate жоғары → сканерлеу
    if f["diff_srv_rate"] > 0.6:
        score += 0.20
    elif f["diff_srv_rate"] > 0.3:
        score += 0.10
    # Аз srv_count → жаңа хосттарды іздеу
    if f["srv_count"] < 10:
        score += 0.20
    # Қысқа немесе нөл duration → тез сканерлеу (0 де жарайды)
    if f["duration"] < 3:
        score += 0.15
    # ICMP протоколы → ping sweep (Probe-тің ең нақты белгісі)
    if f["proto_icmp"]:
        score += 0.35
    # serror_rate аз → DoS емес
    if f["serror_rate"] < 0.1:
        score += 0.10
    # DoS белгілері жоқ болса — Probe ықтималдығы артады
    if not f["flag_s0"] and not f["flag_rej"]:
        score += 0.10
    return max(0.0, min(score, 1.0))


def _score_r2l(f: Dict[str, float]) -> float:
    score = 0.0
    # Кіру сәтсіздіктері → brute force
    if f["num_failed_logins"] > 0:
        score += 0.35
    # Кірілмеген → сәтсіз кіру
    if not f["logged_in"]:
        score += 0.20
    # Ұзақ сессия → тыңшылау
    if f["duration"] > 100:
        score += 0.20
    # Қонақ кіруі
    if f["is_guest_login"]:
        score += 0.25
    return min(score, 1.0)


def _score_u2r(f: Dict[str, float]) -> float:
    score = 0.0
    # root shell ашылған
    if f["root_shell"]:
        score += 0.40
    # su командасы қолданылған
    if f["su_attempted"]:
        score += 0.30
    # root процестері
    if f["num_root"] > 0:
        score += 0.20
    # Файл жасалған → exploit
    if f["num_file_creations"] > 0:
        score += 0.10
    return min(score, 1.0)


# ── Көмекші функциялар ───────────────────────────────────────────────────────

def _safe_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Белгілер сөздігін қауіпсіз float мәндеріне айналдыру."""
    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(raw.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    flag = str(raw.get("flag", "")).upper()
    proto = str(raw.get("protocol_type", "")).lower()

    return {
        "duration":              _f("duration"),
        "src_bytes":             _f("src_bytes"),
        "dst_bytes":             _f("dst_bytes"),
        "count":                 _f("count"),
        "srv_count":             _f("srv_count"),
        "serror_rate":           _f("serror_rate"),
        "srv_serror_rate":       _f("srv_serror_rate"),
        "rerror_rate":           _f("rerror_rate"),
        "dst_host_diff_srv_rate":_f("dst_host_diff_srv_rate"),
        "diff_srv_rate":         _f("diff_srv_rate"),
        "num_failed_logins":     _f("num_failed_logins"),
        "logged_in":             _f("logged_in"),
        "is_guest_login":        _f("is_guest_login"),
        "root_shell":            _f("root_shell"),
        "su_attempted":          _f("su_attempted"),
        "num_root":              _f("num_root"),
        "num_file_creations":    _f("num_file_creations"),
        # Derived boolean flags
        "flag_s0":  "S0"  in flag,
        "flag_rej": "REJ" in flag,
        "proto_icmp": proto == "icmp",
    }


def _build_result(attack_type: str, confidence: float, scores: Dict[str, float]) -> Dict[str, Any]:
    return {
        "type":       attack_type,
        "confidence": confidence,
        "scores":     {k: round(v, 4) for k, v in scores.items()},
        "meta":       ATTACK_META.get(attack_type, ATTACK_META["Unknown"]),
    }


def classify_batch(predictions: list) -> list:
    """
    Batch болжамдар тізімін шабуыл типімен байыту.

    Args:
        predictions: predict_batch() нәтижелері (list of dicts)

    Returns:
        Бастапқы тізімге 'attack_type' кілті қосылған нұсқасы
    """
    enriched = []
    for pred in predictions:
        p = dict(pred)
        if p.get("success") and p.get("prediction") == "Attack":
            features = p.get("features") or {}
            prob = p.get("probability", 0.5)
            p["attack_type"] = classify_attack_type(features, prob)
        else:
            p["attack_type"] = _build_result("Normal", 1.0, {})
        enriched.append(p)
    return enriched
def localize_attack_meta(meta, lang='kk'):
    if lang == 'en':
        return {
            **meta,
            'full_name': meta.get('full_name_en', meta.get('full_name')),
            'description': meta.get('description_en', meta.get('description')),
            'severity': meta.get('severity_en', meta.get('severity')),
        }

    if lang == 'ru':
        severity_map = {
            'Critical': 'Критическая',
            'High': 'Высокая',
            'Medium': 'Средняя',
            'Low': 'Низкая',
            'Unknown': 'Неизвестно',
        }

        name_map = {
            'DDoS': 'Распределённая атака отказа в обслуживании',
            'PortScan': 'Сканирование портов',
            'Bot': 'Ботнет-трафик',
            'DoS Hulk': 'DoS Hulk атака',
            'DoS GoldenEye': 'DoS GoldenEye атака',
            'DoS slowloris': 'DoS slowloris атака',
            'DoS Slowhttptest': 'DoS Slowhttptest атака',
            'FTP-Patator': 'FTP brute force',
            'SSH-Patator': 'SSH brute force',
            'Web Attack Brute Force': 'Web brute force атака',
            'Web Attack XSS': 'XSS веб-атака',
            'Web Attack SQL Injection': 'SQL Injection веб-атака',
            'Infiltration': 'Скрытое проникновение',
            'Heartbleed': 'Уязвимость Heartbleed',
        }

        desc_map = {
            'Many sources overwhelm a target service with traffic.':
                'Множество источников перегружают сервер трафиком.',
            'Reconnaissance activity that probes open ports and services.':
                'Поиск открытых портов и сервисов.',
            'Automated malicious traffic from compromised hosts.':
                'Автоматический вредоносный трафик от заражённых устройств.',
            'High-volume HTTP denial-of-service traffic.':
                'Перегрузка сервера большим количеством HTTP-запросов.',
            'HTTP denial-of-service traffic with randomized requests.':
                'HTTP DoS-атака со случайными запросами.',
            'Slow HTTP connection exhaustion attack.':
                'Медленная атака, истощающая HTTP-соединения.',
            'Slow HTTP request/response denial-of-service pattern.':
                'Медленная HTTP DoS-атака через запросы или ответы.',
            'Repeated FTP login attempts.':
                'Многократные попытки подбора FTP-логина.',
            'Repeated SSH login attempts.':
                'Многократные попытки подбора SSH-логина.',
            'Repeated authentication attempts against a web application.':
                'Многократные попытки подбора логина в веб-приложении.',
            'Malicious script injection attempt against a web application.':
                'Попытка внедрения вредоносного скрипта в веб-приложение.',
            'Attempt to manipulate database queries through user input.':
                'Попытка изменить SQL-запрос через пользовательский ввод.',
            'Suspicious traffic associated with unauthorized internal access.':
                'Подозрительный трафик, связанный с несанкционированным доступом.',
            'Traffic associated with the OpenSSL Heartbleed vulnerability.':
                'Трафик, связанный с уязвимостью OpenSSL Heartbleed.',
        }

        attack_key = None
        for key, attack_meta in ATTACK_META.items():
            if meta is attack_meta or meta == attack_meta:
                attack_key = key
                break

        return {
            **meta,
            'full_name': name_map.get(attack_key, meta.get('full_name')),
            'description': desc_map.get(meta.get('description'), meta.get('description')),
            'severity': severity_map.get(meta.get('severity'), meta.get('severity')),
        }

    severity_map = {
        'Critical': 'Өте жоғары',
        'High': 'Жоғары',
        'Medium': 'Орташа',
        'Low': 'Төмен',
        'Unknown': 'Белгісіз'
    }

    desc_map = {
        'Many sources overwhelm a target service with traffic.':
            'Көптеген көздер серверді трафикпен шамадан тыс жүктейді.',

        'Reconnaissance activity that probes open ports and services.':
            'Ашық порттар мен қызметтерді іздеу әрекеті.',

        'Automated malicious traffic from compromised hosts.':
            'Жұқтырылған құрылғылардан келетін автоматты зиянды трафик.',

        'Repeated SSH login attempts.':
            'SSH логинін қайта-қайта болжау әрекеті.',

        'Repeated FTP login attempts.':
            'FTP логинін қайта-қайта болжау әрекеті.',

        'Attempt to manipulate database queries through user input.':
            'Пайдаланушы енгізуі арқылы SQL шабуылы.',

        'Malicious script injection attempt against a web application.':
            'Веб-бетке зиянды script енгізу әрекеті.',
    }

    return {
        **meta,
        'full_name': meta.get('kk', meta.get('full_name')),
        'description': desc_map.get(
            meta.get('description'),
            meta.get('description')
        ),
        'severity': severity_map.get(
            meta.get('severity'),
            meta.get('severity')
        ),
    }
