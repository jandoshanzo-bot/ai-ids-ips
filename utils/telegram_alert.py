"""
Telegram alert utilities for IDS.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from html import escape

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"


ATTACK_DESCRIPTIONS = {
    "DDoS": {
        "kk": "Серверді шамадан тыс жүктеу үшін көп трафик жіберетін шабуыл.",
        "ru": "Атака большим потоком трафика для перегрузки сервера.",
        "en": "Massive traffic flood to overload a server.",
    },
    "PortScan": {
        "kk": "Ашық порттар мен әлсіз қызметтерді іздеу әрекеті.",
        "ru": "Сканирование портов для поиска открытых сервисов и уязвимостей.",
        "en": "Scanning ports to discover open services and weaknesses.",
    },
    "Bot": {
        "kk": "Ботнет құрамындағы жұқтырылған құрылғы әрекеті.",
        "ru": "Активность заражённого устройства в составе ботнета.",
        "en": "Infected machine communicating as part of a botnet.",
    },
    "DoS Hulk": {
        "kk": "HTTP сұраныстарымен серверді шамадан тыс жүктейтін шабуыл.",
        "ru": "DoS-атака большим количеством HTTP-запросов.",
        "en": "High-volume HTTP flood attack.",
    },
    "Web Attack XSS": {
        "kk": "Веб-бетке зиянды script енгізу әрекеті.",
        "ru": "Попытка внедрения вредоносного скрипта.",
        "en": "Cross-site scripting attempt.",
    },
    "Web Attack SQL Injection": {
        "kk": "SQL код енгізу арқылы дерекқорға шабуыл әрекеті.",
        "ru": "Попытка SQL-инъекции.",
        "en": "SQL injection attempt.",
    },
}


def format_alert_message(
    prediction,
    probability,
    confidence,
    model_name,
    feature_importance=None,
    features=None,
    attack_type_info=None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attack_type = "Unknown"

    if isinstance(attack_type_info, dict):
        attack_type = attack_type_info.get("type") or "Unknown"
        
    if isinstance(attack_type_info, dict) and attack_type_info.get("custom_message"):
        return attack_type_info["custom_message"]

    if prediction == "Attack":
        return (
            "🚨 <b>IDS ЕСКЕРТУ</b>\n\n"
            "Шабуыл анықталды!\n"
            f"Шабуыл түрі: {escape(str(attack_type))}\n"
            f"Сенімділік: {float(confidence) * 100:.1f}%\n"
            f"Уақыты: {timestamp}"
        )

    return (
        "✅ <b>IDS КҮЙІ</b>\n\n"
        "Трафик қалыпты.\n"
        f"Сенімділік: {float(confidence) * 100:.1f}%\n"
        f"Уақыты: {timestamp}"
    )


def send_telegram_alert(
    prediction,
    probability,
    confidence,
    model_name,
    feature_importance=None,
    features=None,
    attack_type_info=None,
    lang="kk",
    force=False,
):
    try:
        if not BOT_TOKEN or not CHAT_ID:
            return {"success": False, "error": "Telegram not configured"}

        if not force and (prediction != "Attack" or float(probability) < ALERT_THRESHOLD):
            return {"success": True, "message": "Skipped"}

        message = format_alert_message(
            prediction,
            probability,
            confidence,
            model_name,
            feature_importance,
            features,
            attack_type_info,
        )

        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }

        response = requests.post(TELEGRAM_API_URL, json=payload, timeout=10)
        return {"success": response.status_code == 200}

    except Exception as exc:
        return {"success": False, "error": str(exc)}


def format_csv_summary_alert(
    filename,
    model_name,
    total_rows,
    attack_rows,
    benign_rows,
    attack_distribution,
    highest_confidence,
    lang="kk",
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lang = lang if lang in ["kk", "ru", "en"] else "kk"

    TEXT = {
        "kk": {
            "title": "🚨 <b>Шабуыл анықталды</b>",
            "file": "Файл",
            "model": "Модель",
            "total": "Жалпы жол",
            "attack": "Шабуыл жолдары",
            "benign": "Қауіпсіз жолдар",
            "types": "Шабуыл түрлері",
            "confidence": "Ең жоғары сенімділік",
            "time": "Уақыты",
        },
        "ru": {
            "title": "🚨 <b>Обнаружена атака</b>",
            "file": "Файл",
            "model": "Модель",
            "total": "Всего строк",
            "attack": "Строк атак",
            "benign": "Безопасных строк",
            "types": "Типы атак",
            "confidence": "Макс уверенность",
            "time": "Время",
        },
        "en": {
            "title": "🚨 <b>Attack Detected</b>",
            "file": "File",
            "model": "Model",
            "total": "Total rows",
            "attack": "Attack rows",
            "benign": "Benign rows",
            "types": "Attack types",
            "confidence": "Highest confidence",
            "time": "Time",
        },
    }

    t = TEXT[lang]

    lines = []
    for item in attack_distribution[:5]:
        attack_type = str(item.get("type", "Unknown"))
        count = int(item.get("count", 0))
        desc = ATTACK_DESCRIPTIONS.get(attack_type, {}).get(lang, "")
        lines.append(
            f"• <b>{escape(attack_type)}</b>: {count}\n"
            f"  {escape(desc)}"
        )

    attack_text = "\n".join(lines) if lines else "• Unknown"

    return (
        f"{t['title']}\n\n"
        f"<b>{t['file']}:</b> {escape(str(filename))}\n"
        f"<b>{t['model']}:</b> {escape(str(model_name))}\n\n"
        f"<b>{t['total']}:</b> {int(total_rows)}\n"
        f"<b>{t['attack']}:</b> {int(attack_rows)}\n"
        f"<b>{t['benign']}:</b> {int(benign_rows)}\n\n"
        f"<b>{t['types']}:</b>\n"
        f"{attack_text}\n\n"
        f"<b>{t['confidence']}:</b> {float(highest_confidence) * 100:.1f}%\n"
        f"<b>{t['time']}:</b> {timestamp}"
    )


def send_telegram_summary_alert(
    filename,
    model_name,
    total_rows,
    attack_rows,
    benign_rows,
    attack_distribution,
    highest_confidence,
    lang="kk",
):
    try:
        if int(attack_rows) <= 0:
            return {"success": True}

        if not BOT_TOKEN or not CHAT_ID:
            return {"success": False, "error": "Telegram not configured"}

        message = format_csv_summary_alert(
            filename=filename,
            model_name=model_name,
            total_rows=total_rows,
            attack_rows=attack_rows,
            benign_rows=benign_rows,
            attack_distribution=attack_distribution,
            highest_confidence=highest_confidence,
            lang=lang,
        )

        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }

        response = requests.post(TELEGRAM_API_URL, json=payload, timeout=10)

        return {"success": response.status_code == 200}

    except Exception as exc:
     return {"success": False, "error": str(exc)}


def test_telegram_connection():
    """Validate Telegram bot configuration."""
    try:
        if not BOT_TOKEN:
            return {"success": False, "bot_info": None, "error": "Бот токені бапталмаған"}

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return {
                "success": False,
                "bot_info": None,
                "error": f"HTTP {response.status_code}",
            }

        result = response.json()

        if result.get("ok"):
            return {
                "success": True,
                "bot_info": result["result"],
                "error": None,
            }

        return {
            "success": False,
            "bot_info": None,
            "error": result.get("description", "Unknown error"),
        }

    except Exception as exc:
        return {"success": False, "bot_info": None, "error": str(exc)}


def get_bot_info():
    """Return bot profile info."""
    result = test_telegram_connection()

    if result["success"]:
        bot = result["bot_info"]

        return {
            "name": bot.get("first_name", "Unknown"),
            "username": bot.get("username", "Unknown"),
            "id": bot.get("id", "Unknown"),
            "is_active": True,
        }

    return {
        "name": "Бапталмаған",
        "username": "Unknown",
        "id": "Unknown",
        "is_active": False,
        "error": result.get("error"),
    }