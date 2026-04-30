"""
Intrusion Detection System (IDS) Web Application

Flask + Machine Learning + Telegram Alerts
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from functools import wraps
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from models.predict import predict_batch
from models.predict_cicids import (
    CLASSES as CICIDS_CLASSES,
    FEATURE_NAMES as CICIDS_FEATURE_NAMES,
    predict_cicids_dataframe,
)
from models.predict_iiot2025 import (
    CLASSES as IIOT2025_CLASSES,
    FEATURE_NAMES as IIOT2025_FEATURE_NAMES,
    IIOT2025_MODEL_NAME,
    predict_iiot2025_dataframe,
)
from utils.attack_classifier import ATTACK_META, localize_attack_meta
from models.train_models import MODEL_DIR
from utils.telegram_alert import send_telegram_alert, send_telegram_summary_alert, get_bot_info
from utils.database import db_manager
from utils.attack_classifier import ATTACK_META

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Flask app setup
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 16MB

ALLOWED_EXTENSIONS = {'csv', 'json'}

CICIDS_MODEL_ID = 'cicids2017_xgboost'
CICIDS_MODEL_NAME = 'CICIDS2017 XGBoost'
IIOT2025_MODEL_ID = 'iiot2025'
SUPPORTED_UPLOAD_MODELS = {'cicids', IIOT2025_MODEL_ID}
BENIGN_LABEL = 'BENIGN'
BENIGN_HISTORY_LIMIT = 100
CICIDS_ATTACK_TYPES = [
    'DDoS',
    'PortScan',
    'Bot',
    'DoS Hulk',
    'DoS GoldenEye',
    'DoS slowloris',
    'DoS Slowhttptest',
    'FTP-Patator',
    'SSH-Patator',
    'Web Attack Brute Force',
    'Web Attack XSS',
    'Web Attack SQL Injection',
    'Infiltration',
    'Heartbleed',
]
ATTACK_TYPE_ALIASES = {
    'Web Attack � Brute Force': 'Web Attack Brute Force',
    'Web Attack � XSS': 'Web Attack XSS',
    'Web Attack � Sql Injection': 'Web Attack SQL Injection',
    'Web Attack Sql Injection': 'Web Attack SQL Injection',
}

# Global predictor status
predictor = {
    'is_loaded': True,
    'model_name': CICIDS_MODEL_NAME,
    'features': len(CICIDS_FEATURE_NAMES),
    'classes': CICIDS_CLASSES,
}

SUPPORTED_LANGUAGES = ('kk', 'ru', 'en')
DEFAULT_LANGUAGE = 'kk'

TRANSLATIONS = {
    'kk': {
        'base.title_default': 'IDS - Желіге енуді анықтау',
        'base.nav.home': 'Басты бет',
        'base.nav.predict': 'Болжау',
        'base.nav.upload': 'Файл жүктеу',
        'base.nav.history': 'Тарих',
        'base.nav.attacks': 'Шабуылдар',
        'base.nav.statistics': 'Статистика',
        'base.nav.settings': 'Баптаулар',
        'base.nav.logout': 'Шығу',
        'base.nav.login': 'Кіру',
        'base.nav.register': 'Тіркелу',
        'base.theme_toggle': 'Теманы ауыстыру',
        'base.language': 'Тіл',
        'base.lang.kk': 'Қазақша',
        'base.lang.ru': 'Русский',
        'base.lang.en': 'English',
        'base.footer_tagline': 'AI Powered Security',
        'base.footer.system_online': 'Жүйе белсенді',
        'login.title': 'Кіру',
        'login.welcome': 'Қош келдіңіз',
        'login.subtitle': 'IDS жүйесіне кіру',
        'login.username': 'Пайдаланушы аты',
        'login.username_placeholder': 'Пайдаланушы атын енгізіңіз',
        'login.password': 'Құпия сөз',
        'login.password_placeholder': 'Құпия сөзді енгізіңіз',
        'login.submit': 'Кіру',
        'login.no_account': 'Аккаунт жоқ па?',
        'login.register_link': 'Тіркелу',
        'register.title': 'Тіркелу',
        'register.new_account': 'Жаңа аккаунт',
        'register.subtitle': 'IDS жүйесіне тіркелу',
        'register.username': 'Пайдаланушы аты',
        'register.username_placeholder': 'Username (3+ таңба)',
        'register.password': 'Құпия сөз',
        'register.password_placeholder': 'Құпия сөз (6+ таңба)',
        'register.confirm_password': 'Құпия сөзді қайталау',
        'register.confirm_placeholder': 'Құпия сөзді растаңыз',
        'register.submit': 'Тіркелу',
        'register.has_account': 'Аккаунтыңыз бар ма?',
        'register.login_link': 'Кіру',
        'error.title': 'Қате {code}',
        'error.description': 'Сіз іздеген бет табылмады немесе қате орын алды.',
        'error.back_home': 'Басты бетке оралу',
        'error.not_found': 'Бет табылмады',
        'error.server': 'Ішкі сервер қатесі',
        'index.title': 'IDS - Желіге енуді анықтау жүйесі',
        'index.hero_title': 'Жасанды интеллект негізіндегі IDS',
        'index.hero_subtitle': 'Машиналық оқыту негізіндегі кибершабуылдарды анықтау жүйесі',
        'index.hero_predict': 'Болжам жасау',
        'index.hero_upload': 'Файл жүктеу',
        'index.stats.total': 'Талдаулар',
        'index.stats.attacks': 'Шабуылдар',
        'index.stats.normal': 'Қауіпсіз',
        'index.stats.alerts': 'Ескертулер',
        'index.features.title': 'Негізгі мүмкіндіктер',
        'index.features.ml_title': 'ML модельдері',
        'index.features.ml_desc': 'XGBoost, Random Forest',
        'index.features.scaler_title': 'Тұрақты әдіс',
        'index.features.scaler_desc': 'RobustScaler',
        'index.features.telegram_title': 'Telegram ескертулері',
        'index.features.telegram_desc': 'Автоматты хабарламалар',
        'index.features.history_title': 'Тарих сақтау',
        'index.features.history_desc': 'SQLite дерекқоры',
        'index.features.parameters': 'Негізгі параметрлер',
        'index.features.more_params': 'және тағы 30+ параметр...',
        'index.models.title': 'Модельдер',
        'index.models.active': 'Белсенді',
        'index.models.metric.accuracy': 'Дәлдік',
        'index.models.metric.precision': 'Нақтылық',
        'index.models.metric.recall': 'Қамту',
        'index.models.metric.f1': 'F1-көрсеткіш',
        'index.models.loading': 'Сақталған метрикалар жоқ',
        'index.models.full_statistics': 'Толық статистика',
        'index.quick.title': 'Жылдам әрекеттер',
        'index.quick.manual_title': 'Қолмен енгізу',
        'index.quick.manual_desc': 'Жеке мәндермен талдау',
        'index.quick.csv_title': 'CSV жүктеу',
        'index.quick.csv_desc': 'Файлды жаппай талдау',
        'index.quick.api_desc': 'Интеграция құжаттамасы',
        'index.telegram.title': 'Telegram боты',
        'index.telegram.not_configured': 'Бот бапталмаған',
        'index.telegram.settings': 'Баптау',
        'api.auth_required': 'Аутентификация қажет',
        'flash.login_required': 'Алдымен жүйеге кіріңіз.',
        'flash.register_username_len': 'Пайдаланушы аты кемінде 3 таңба болуы керек.',
        'flash.register_password_len': 'Құпия сөз кемінде 6 таңба болуы керек.',
        'flash.register_password_mismatch': 'Құпия сөздер сәйкес келмейді.',
        'flash.register_failed': 'Тіркелу сәтсіз аяқталды.',
        'flash.register_success': 'Тіркелу сәтті аяқталды.',
        'flash.login_invalid': 'Пайдаланушы аты немесе құпия сөз қате.',
        'flash.login_success': 'Кіру сәтті орындалды.',
        'flash.logout': 'Сіз жүйеден шықтыңыз.',
        'flash.predict_failed': 'Болжам қатесі: {error}',
        'flash.error': 'Қате: {error}',
        'flash.file_not_selected': 'Файл таңдалмады',
        'flash.file_read_error': 'Файлды өңдеу қатесі: {error}',
        'flash.invalid_file_type': 'Қолдау көрсетілмейтін файл форматы. Тек CSV және JSON.',
    },
    'ru': {
        'base.title_default': 'IDS - Обнаружение сетевых вторжений',
        'base.nav.home': 'Главная',
        'base.nav.predict': 'Прогноз',
        'base.nav.upload': 'Загрузка файла',
        'base.nav.history': 'История',
        'base.nav.attacks': 'Атаки',
        'base.nav.statistics': 'Статистика',
        'base.nav.settings': 'Настройки',
        'base.nav.logout': 'Выйти',
        'base.nav.login': 'Войти',
        'base.nav.register': 'Регистрация',
        'base.theme_toggle': 'Сменить тему',
        'base.language': 'Язык',
        'base.lang.kk': 'Қазақша',
        'base.lang.ru': 'Русский',
        'base.lang.en': 'English',
        'base.footer_tagline': 'AI Powered Security',
        'base.footer.system_online': 'Система активна',
        'login.title': 'Вход',
        'login.welcome': 'Добро пожаловать',
        'login.subtitle': 'Вход в систему IDS',
        'login.username': 'Имя пользователя',
        'login.username_placeholder': 'Введите имя пользователя',
        'login.password': 'Пароль',
        'login.password_placeholder': 'Введите пароль',
        'login.submit': 'Войти',
        'login.no_account': 'Нет аккаунта?',
        'login.register_link': 'Регистрация',
        'register.title': 'Регистрация',
        'register.new_account': 'Новый аккаунт',
        'register.subtitle': 'Регистрация в системе IDS',
        'register.username': 'Имя пользователя',
        'register.username_placeholder': 'Username (3+ символа)',
        'register.password': 'Пароль',
        'register.password_placeholder': 'Пароль (6+ символов)',
        'register.confirm_password': 'Повторите пароль',
        'register.confirm_placeholder': 'Подтвердите пароль',
        'register.submit': 'Зарегистрироваться',
        'register.has_account': 'Уже есть аккаунт?',
        'register.login_link': 'Войти',
        'error.title': 'Ошибка {code}',
        'error.description': 'Страница не найдена или произошла ошибка.',
        'error.back_home': 'На главную',
        'error.not_found': 'Страница не найдена',
        'error.server': 'Внутренняя ошибка сервера',
        'index.title': 'IDS - Система обнаружения вторжений',
        'index.hero_title': 'IDS на базе искусственного интеллекта',
        'index.hero_subtitle': 'Система обнаружения кибератак на основе машинного обучения',
        'index.hero_predict': 'Сделать прогноз',
        'index.hero_upload': 'Загрузить файл',
        'index.stats.total': 'Анализы',
        'index.stats.attacks': 'Атаки',
        'index.stats.normal': 'Безопасно',
        'index.stats.alerts': 'Алерты',
        'index.features.title': 'Ключевые возможности',
        'index.features.ml_title': 'ML модели',
        'index.features.ml_desc': 'XGBoost, Random Forest',
        'index.features.scaler_title': 'Устойчивый метод',
        'index.features.scaler_desc': 'RobustScaler',
        'index.features.telegram_title': 'Уведомления Telegram',
        'index.features.telegram_desc': 'Автоматические сообщения',
        'index.features.history_title': 'Хранение истории',
        'index.features.history_desc': 'База данных SQLite',
        'index.features.parameters': 'Основные параметры',
        'index.features.more_params': 'и еще 30+ параметров...',
        'index.models.title': 'Модели',
        'index.models.active': 'Активна',
        'index.models.metric.accuracy': 'Точность',
        'index.models.metric.precision': 'Precision',
        'index.models.metric.recall': 'Recall',
        'index.models.metric.f1': 'F1-показатель',
        'index.models.loading': 'Сохраненных метрик нет',
        'index.models.full_statistics': 'Полная статистика',
        'index.quick.title': 'Быстрые действия',
        'index.quick.manual_title': 'Ручной ввод',
        'index.quick.manual_desc': 'Анализ с индивидуальными значениями',
        'index.quick.csv_title': 'Загрузка CSV',
        'index.quick.csv_desc': 'Массовый анализ файла',
        'index.quick.api_desc': 'Документация по интеграции',
        'index.telegram.title': 'Telegram бот',
        'index.telegram.not_configured': 'Бот не настроен',
        'index.telegram.settings': 'Настроить',
        'api.auth_required': 'Требуется аутентификация',
        'flash.login_required': 'Сначала войдите в систему.',
        'flash.register_username_len': 'Имя пользователя должно быть не менее 3 символов.',
        'flash.register_password_len': 'Пароль должен быть не менее 6 символов.',
        'flash.register_password_mismatch': 'Пароли не совпадают.',
        'flash.register_failed': 'Регистрация завершилась ошибкой.',
        'flash.register_success': 'Регистрация прошла успешно.',
        'flash.login_invalid': 'Неверное имя пользователя или пароль.',
        'flash.login_success': 'Вход выполнен успешно.',
        'flash.logout': 'Вы вышли из системы.',
        'flash.predict_failed': 'Ошибка прогноза: {error}',
        'flash.error': 'Ошибка: {error}',
        'flash.file_not_selected': 'Файл не выбран',
        'flash.file_read_error': 'Ошибка обработки файла: {error}',
        'flash.invalid_file_type': 'Неподдерживаемый формат файла. Только CSV и JSON.',
    },
    'en': {
        'base.title_default': 'IDS - Network Intrusion Detection',
        'base.nav.home': 'Home',
        'base.nav.predict': 'Predict',
        'base.nav.upload': 'Upload File',
        'base.nav.history': 'History',
        'base.nav.attacks': 'Attacks',
        'base.nav.statistics': 'Statistics',
        'base.nav.settings': 'Settings',
        'base.nav.logout': 'Logout',
        'base.nav.login': 'Login',
        'base.nav.register': 'Register',
        'base.theme_toggle': 'Toggle theme',
        'base.language': 'Language',
        'base.lang.kk': 'Қазақша',
        'base.lang.ru': 'Русский',
        'base.lang.en': 'English',
        'base.footer_tagline': 'AI Powered Security',
        'base.footer.system_online': 'System online',
        'login.title': 'Login',
        'login.welcome': 'Welcome back',
        'login.subtitle': 'Sign in to IDS',
        'login.username': 'Username',
        'login.username_placeholder': 'Enter username',
        'login.password': 'Password',
        'login.password_placeholder': 'Enter password',
        'login.submit': 'Login',
        'login.no_account': "Don't have an account?",
        'login.register_link': 'Register',
        'register.title': 'Register',
        'register.new_account': 'Create account',
        'register.subtitle': 'Register in IDS',
        'register.username': 'Username',
        'register.username_placeholder': 'Username (3+ chars)',
        'register.password': 'Password',
        'register.password_placeholder': 'Password (6+ chars)',
        'register.confirm_password': 'Confirm password',
        'register.confirm_placeholder': 'Repeat password',
        'register.submit': 'Create account',
        'register.has_account': 'Already have an account?',
        'register.login_link': 'Login',
        'error.title': 'Error {code}',
        'error.description': 'The page was not found or an error occurred.',
        'error.back_home': 'Back to home',
        'error.not_found': 'Page not found',
        'error.server': 'Internal server error',
        'index.title': 'IDS - Intrusion Detection System',
        'index.hero_title': 'AI-Powered IDS',
        'index.hero_subtitle': 'Machine-learning based cyberattack detection system',
        'index.hero_predict': 'Run prediction',
        'index.hero_upload': 'Upload file',
        'index.stats.total': 'Analyses',
        'index.stats.attacks': 'Attacks',
        'index.stats.normal': 'Safe',
        'index.stats.alerts': 'Alerts',
        'index.features.title': 'Core features',
        'index.features.ml_title': 'ML Models',
        'index.features.ml_desc': 'XGBoost, Random Forest',
        'index.features.scaler_title': 'Robust method',
        'index.features.scaler_desc': 'RobustScaler',
        'index.features.telegram_title': 'Telegram alerts',
        'index.features.telegram_desc': 'Automated notifications',
        'index.features.history_title': 'History storage',
        'index.features.history_desc': 'SQLite database',
        'index.features.parameters': 'Key parameters',
        'index.features.more_params': 'and 30+ more parameters...',
        'index.models.title': 'Models',
        'index.models.active': 'Active',
        'index.models.metric.accuracy': 'Accuracy',
        'index.models.metric.precision': 'Precision',
        'index.models.metric.recall': 'Recall',
        'index.models.metric.f1': 'F1-score',
        'index.models.loading': 'No saved metrics',
        'index.models.full_statistics': 'Full statistics',
        'index.quick.title': 'Quick actions',
        'index.quick.manual_title': 'Manual input',
        'index.quick.manual_desc': 'Analyze custom values',
        'index.quick.csv_title': 'Upload CSV',
        'index.quick.csv_desc': 'Bulk file analysis',
        'index.quick.api_desc': 'Integration docs',
        'index.telegram.title': 'Telegram bot',
        'index.telegram.not_configured': 'Bot is not configured',
        'index.telegram.settings': 'Configure',
        'api.auth_required': 'Authentication required',
        'flash.login_required': 'Please sign in first.',
        'flash.register_username_len': 'Username must be at least 3 characters.',
        'flash.register_password_len': 'Password must be at least 6 characters.',
        'flash.register_password_mismatch': 'Passwords do not match.',
        'flash.register_failed': 'Registration failed.',
        'flash.register_success': 'Registration completed successfully.',
        'flash.login_invalid': 'Invalid username or password.',
        'flash.login_success': 'Logged in successfully.',
        'flash.logout': 'You are logged out.',
        'flash.predict_failed': 'Prediction failed: {error}',
        'flash.error': 'Error: {error}',
        'flash.file_not_selected': 'No file selected',
        'flash.file_read_error': 'File processing error: {error}',
        'flash.invalid_file_type': 'Unsupported file format. Only CSV and JSON are allowed.',
    },
}

def get_current_language():
    """Return active language code from user session."""
    lang = session.get('lang', DEFAULT_LANGUAGE)
    if lang in SUPPORTED_LANGUAGES:
        return lang
    return DEFAULT_LANGUAGE


def t(key, **kwargs):
    """Translate key for the current session language with fallback."""
    lang = get_current_language()
    text = (
        TRANSLATIONS.get(lang, {}).get(key)
        or TRANSLATIONS[DEFAULT_LANGUAGE].get(key)
        or key
    )
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text


def ts(kk_text, ru_text, en_text):
    """Select an inline Kazakh/Russian/English template string."""
    lang = get_current_language()
    if lang == 'ru':
        return ru_text
    if lang == 'en':
        return en_text
    return kk_text


@app.context_processor
def inject_i18n():
    """Expose translation helper and locale info to all templates."""
    return {
        't': t,
        'ts': ts,
        'current_lang': get_current_language(),
        'supported_languages': SUPPORTED_LANGUAGES,
    }


def init_app():
    """Initialize application state."""
    global predictor

    logger.info("IDS application initializing...")
    predictor = {
        'is_loaded': True,
        'model_name': CICIDS_MODEL_NAME,
        'features': len(CICIDS_FEATURE_NAMES),
        'classes': CICIDS_CLASSES,
    }
    logger.info("CICIDS2017 model ready")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_plot_urls():
    """Trained model plot URLs for statistics page."""
    plot_dir = MODEL_DIR / 'plots'
    plot_files = {
        'roc_curve': 'xgboost_roc_curve.png',
        'feature_importance': 'xgboost_feature_importance.png',
        'accuracy_comparison': 'model_accuracy_comparison.png',
        'attack_distribution': 'attack_distribution_pie.png',
        'confusion_matrix': 'xgboost_confusion_matrix.png',
    }
    urls = {}
    for key, filename in plot_files.items():
        path = plot_dir / filename
        if path.exists() and path.is_file():
            urls[key] = url_for('model_plot', filename=filename) + f"?v={int(path.stat().st_mtime)}"
    return urls


def get_cicids_models():
    return [{
        "name": "cicids2017_xgboost",
        "display_name": "CICIDS2017 XGBoost",
        "status": "loaded",

        "metrics": {
            "accuracy": 0.9988,
            "precision": 0.9986,
            "recall": 0.9989,
            "f1_score": 0.9988
        },

        "dataset": "CICIDS2017",
        "classes": 15,
        "features": len(CICIDS_FEATURE_NAMES),
    }, {
        "name": IIOT2025_MODEL_ID,
        "display_name": IIOT2025_MODEL_NAME,
        "status": "loaded",
        "metrics": {
            "accuracy": 0.9565,
            "precision": 0.9573,
            "recall": 0.9565,
            "f1_score": 0.9565
        },
        "dataset": "CIC_IIoT_2025",
        "classes": len(IIOT2025_CLASSES),
        "features": len(IIOT2025_FEATURE_NAMES),
    }]


def get_cicids_sample_values(limit=20):
    """Default numeric values for the CICIDS manual prediction form."""
    return {feature: 0 for feature in CICIDS_FEATURE_NAMES[:limit]}


def normalize_cicids_result(result, row=None):
    """Map CICIDS class-label output into the existing UI/API shape."""
    raw_label = str(result.get('prediction', 'Unknown'))
    is_attack = bool(result.get('is_attack', raw_label.upper() != BENIGN_LABEL))
    attack_probability = float(result.get('attack_probability', result.get('confidence', 0.0)) or 0.0)
    confidence = float(result.get('confidence', attack_probability) or 0.0)
    attack_type = canonical_attack_type(raw_label) if is_attack else 'None'
    class_probabilities = result.get('class_probabilities') or {}

    normalized = {
        'success': True,
        'prediction': 'Attack' if is_attack else 'Normal',
        'raw_prediction': raw_label,
        'is_attack': is_attack,
        'probability': attack_probability,
        'attack_probability': attack_probability,
        'confidence': confidence,
        'model_name': result.get('model_name', CICIDS_MODEL_NAME),
        'attack_type': attack_type,
        'attack_type_confidence': confidence if is_attack else None,
        'class_probabilities': class_probabilities,
    }
    if row is not None:
        normalized['row'] = row
    if is_attack:
        normalized['attack_type_info'] = {
            'type': attack_type,
            'confidence': confidence,
            'scores': {
                str(key): float(value)
                for key, value in class_probabilities.items()
            } if isinstance(class_probabilities, dict) else {},
            'meta': ATTACK_META.get(attack_type, ATTACK_META.get('Unknown', {})),
        }
    return normalized


def normalize_iiot2025_result(result, row=None):
    """Map IIoT2025 class labels into the existing upload/API shape."""
    raw_label = str(result.get('prediction', 'Unknown'))
    confidence = float(result.get('confidence', 0.0) or 0.0)
    class_probabilities = result.get('class_probabilities') or {}

    normalized = {
        'success': True,
        'prediction': 'Attack',
        'raw_prediction': raw_label,
        'is_attack': True,
        'probability': confidence,
        'attack_probability': float(result.get('attack_probability', confidence) or 0.0),
        'confidence': confidence,
        'model_name': result.get('model_name', IIOT2025_MODEL_NAME),
        'attack_type': raw_label,
        'attack_type_confidence': confidence,
        'class_probabilities': class_probabilities,
        'attack_type_info': {
            'type': raw_label,
            'confidence': confidence,
            'scores': {
                str(key): float(value)
                for key, value in class_probabilities.items()
            } if isinstance(class_probabilities, dict) else {},
            'meta': ATTACK_META.get(raw_label, ATTACK_META.get('Unknown', {})),
        },
    }
    if row is not None:
        normalized['row'] = row
    return normalized


def build_prediction_summary(results):
    raw_classes = [str(result.get('raw_prediction') or result.get('prediction') or 'Unknown') for result in results]
    class_counts = {}
    for label in raw_classes:
        class_counts[label] = class_counts.get(label, 0) + 1
    return {
        'detected_classes': sorted(class_counts.keys()),
        'class_counts': class_counts,
        'total_rows': len(results),
    }


def save_cicids_prediction(result, features=None):
    """Persist one normalized CICIDS result to history."""
    return db_manager.save_prediction(
        model_name=result.get('model_name', CICIDS_MODEL_NAME),
        prediction=result['prediction'],
        probability=result['attack_probability'],
        confidence=result['confidence'],
        features=features,
        alert_sent=False,
        attack_type=result['attack_type'] if result.get('is_attack') else None,
        attack_type_confidence=result['confidence'] if result.get('is_attack') else None,
    )


def get_attack_meta(attack_type):
    """Return metadata for CICIDS/KDD attack labels."""
    attack_type = canonical_attack_type(attack_type or 'Unknown')
    return ATTACK_META.get(attack_type, ATTACK_META.get('Unknown', {}))


def canonical_attack_type(attack_type):
    """Normalize CICIDS label-encoder variants into display/storage labels."""
    attack_type = str(attack_type or 'Unknown').strip()
    if attack_type.startswith('Web Attack') and 'Brute Force' in attack_type:
        return 'Web Attack Brute Force'
    if attack_type.startswith('Web Attack') and 'XSS' in attack_type.upper():
        return 'Web Attack XSS'
    if attack_type.startswith('Web Attack') and 'SQL' in attack_type.upper():
        return 'Web Attack SQL Injection'
    return ATTACK_TYPE_ALIASES.get(attack_type, attack_type)


def build_attack_catalog(known_types=None):
    known_types = [a for a in (known_types or []) if a]

    ordered = []
    seen = set()

    for attack_type in CICIDS_ATTACK_TYPES + known_types:
        if attack_type in seen or attack_type in {'None', 'Normal', 'BENIGN'}:
            continue

        seen.add(attack_type)

        meta = get_attack_meta(attack_type)
        meta = localize_attack_meta(meta, get_current_language())

        ordered.append({
            'type': attack_type,
            'full_name': meta.get('full_name', attack_type),
            'description': meta.get('description', ''),
            'severity': meta.get('severity', ''),
            'emoji': meta.get('emoji', '❓'),
            'color': meta.get('color', '#6b7280'),
            'badge': meta.get('badge', 'secondary'),
        })

    return ordered


def attach_attack_meta(rows):
    for row in rows:
        attack_type = canonical_attack_type(row.get('attack_type') or 'Unknown')

        row['attack_type_display'] = attack_type

        meta = get_attack_meta(attack_type)
        meta = localize_attack_meta(meta, get_current_language())

        row['attack_type_meta'] = meta

    return rows


def normalize_attack_type_stats(stats):
    """Canonicalize and merge attack type aggregate rows for display."""
    merged = {}
    for row in (stats.get('by_type', []) if stats else []):
        attack_type = canonical_attack_type(row.get('attack_type'))
        count = int(row.get('count') or 0)
        avg_probability = float(row.get('avg_probability') or 0.0)
        current = merged.setdefault(attack_type, {'attack_type': attack_type, 'count': 0, 'weighted_probability': 0.0})
        current['count'] += count
        current['weighted_probability'] += avg_probability * count

    by_type = []
    for row in merged.values():
        count = row['count']
        by_type.append({
            'attack_type': row['attack_type'],
            'count': count,
            'avg_probability': round(row['weighted_probability'] / count, 1) if count else 0.0,
        })
    by_type.sort(key=lambda item: item['count'], reverse=True)

    timeline_merged = {}
    for row in (stats.get('timeline', []) if stats else []):
        attack_type = canonical_attack_type(row.get('attack_type'))
        key = (row.get('hour'), attack_type)
        current = timeline_merged.setdefault(key, {'hour': row.get('hour'), 'attack_type': attack_type, 'count': 0})
        current['count'] += int(row.get('count') or 0)

    timeline = sorted(timeline_merged.values(), key=lambda item: (item.get('hour') or '', item.get('attack_type') or ''))
    return {'by_type': by_type, 'timeline': timeline}


def build_attack_type_info(result):
    """Build normalized attack-type payload from model prediction result."""
    if result.get('prediction') != 'Attack':
        return None

    class_probabilities = result.get('class_probabilities') or {}
    attack_type = result.get('attack_type')
    if attack_type in (None, '', 'None', 'Normal'):
        candidates = {
            key: float(value)
            for key, value in class_probabilities.items()
            if key in {'DoS', 'Probe', 'R2L', 'U2R'}
        }
        attack_type = max(candidates, key=candidates.get) if candidates else 'Unknown'

    attack_confidence = result.get('attack_type_confidence')
    if attack_confidence is None:
        attack_confidence = class_probabilities.get(attack_type, result.get('confidence', 0.0))

    return {
        'type': attack_type,
        'confidence': float(attack_confidence or 0.0),
        'scores': {
            str(key): float(value)
            for key, value in class_probabilities.items()
        } if isinstance(class_probabilities, dict) else {},
        'meta': ATTACK_META.get(attack_type, ATTACK_META['Unknown']),
    }


# WEB ROUTES

def _is_safe_next_url(target):
    """Allow only relative redirects inside this app."""
    if not target:
        return False
    parsed = urlparse(target)
    return parsed.scheme == '' and parsed.netloc == '' and target.startswith('/')

@app.route('/set-language/<lang>', methods=['GET', 'POST'])
def set_language(lang):
    """Persist selected language in user session."""
    if lang not in SUPPORTED_LANGUAGES:
        abort(400)

    session['lang'] = lang
    next_url = request.values.get('next', '')
    if _is_safe_next_url(next_url):
        return redirect(next_url)
    return redirect(url_for('index'))


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if session.get('user_id'):
            return view_func(*args, **kwargs)

        if request.path.startswith('/api/'):
            return jsonify({'success': False, 'error': t('api.auth_required')}), 401

        flash(t('flash.login_required'), 'warning')
        return redirect(url_for('login', next=request.path))

    return wrapped_view


@app.route('/')
def index():
    stats = db_manager.get_statistics()
    
    models = get_cicids_models()
    
    bot_info = get_bot_info()
    
    return render_template('index.html', 
                         stats=stats, 
                         models=models,
                         bot_info=bot_info,
                         feature_names=CICIDS_FEATURE_NAMES[:10])


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if session.get('user_id'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''

        if len(username) < 3:
            flash(t('flash.register_username_len'), 'error')
            return render_template('register.html', username=username)

        if len(password) < 6:
            flash(t('flash.register_password_len'), 'error')
            return render_template('register.html', username=username)

        if password != confirm_password:
            flash(t('flash.register_password_mismatch'), 'error')
            return render_template('register.html', username=username)

        created = db_manager.create_user(
            username=username,
            password_hash=generate_password_hash(password)
        )
        if not created.get('success'):
            flash(created.get('error', t('flash.register_failed')), 'error')
            return render_template('register.html', username=username)

        session['user_id'] = created['user_id']
        session['username'] = username
        flash(t('flash.register_success'), 'success')
        return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    if session.get('user_id'):
        return redirect(url_for('index'))

    next_url = request.args.get('next', '')
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        next_url = request.form.get('next', next_url)

        user = db_manager.get_user_by_username(username)
        if not user or not check_password_hash(user['password_hash'], password):
            flash(t('flash.login_invalid'), 'error')
            return render_template('login.html', username=username, next_url=next_url)

        session['user_id'] = user['id']
        session['username'] = user['username']
        flash(t('flash.login_success'), 'success')

        if _is_safe_next_url(next_url):
            return redirect(next_url)
        return redirect(url_for('index'))

    return render_template('login.html', next_url=next_url)


@app.route('/logout')
def logout():
    """User logout."""
    session.clear()
    flash(t('flash.logout'), 'info')
    return redirect(url_for('index'))


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict_page():
    if request.method == 'POST':
        try:
            data = {}
            for feature in CICIDS_FEATURE_NAMES[:20]:
                value = request.form.get(feature, '0')
                data[feature] = float(value) if value else 0
            
            cicids_result = predict_cicids_dataframe(pd.DataFrame([data]))[0]
            result = normalize_cicids_result(cicids_result)
            
            if result['success']:
                attack_type_info = result.get('attack_type_info')
                prediction_id = None
                if result['is_attack']:
                    prediction_id = save_cicids_prediction(result, features=data)
                
                alert_result = None
                if result['prediction'] == 'Attack' and result['attack_probability'] > 0.7:
                    alert_result = send_telegram_alert(
                        prediction=result['prediction'],
                        probability=result['attack_probability'],
                        confidence=result['confidence'],
                        model_name=result['model_name'],
                        features=data,
                        attack_type_info=attack_type_info,
                    )
                    
                    if prediction_id and alert_result and alert_result.get('success'):
                        db_manager.save_alert(prediction_id, 
                                            alert_result.get('message', 'Alert sent'))
                
                return render_template('predict.html',
                                     result=result,
                                     input_data=data,
                                     alert_result=alert_result,
                                     attack_meta=ATTACK_META,
                                     feature_names=CICIDS_FEATURE_NAMES[:20],
                                     sample_values=get_cicids_sample_values())
            else:
                flash(
                    t('flash.predict_failed', error=result.get('error', t('flash.error', error='Unknown'))),
                    'error'
                )
                
        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            flash(t('flash.error', error=str(e)), 'error')
    
    return render_template('predict.html', 
                         feature_names=CICIDS_FEATURE_NAMES[:20],
                         sample_values=get_cicids_sample_values())


@app.route('/history')
@login_required
def history():
    predictions = attach_attack_meta(db_manager.get_prediction_history(limit=100))
    return render_template('history.html', predictions=predictions, attack_catalog=build_attack_catalog())


@app.route('/statistics')
@login_required
def statistics():
    stats = db_manager.get_statistics()
    models = get_cicids_models()
    recent_attacks = attach_attack_meta(db_manager.get_recent_attacks(limit=10))
    attack_type_stats = normalize_attack_type_stats(db_manager.get_attack_type_stats())
    
    return render_template('statistics.html',
                         stats=stats,
                         models=models,
                         recent_attacks=recent_attacks,
                         attack_type_stats=attack_type_stats)


@app.route('/attacks')
@login_required
def attacks_page():
    attack_type_stats = normalize_attack_type_stats(db_manager.get_attack_type_stats())
    recent_attacks = attach_attack_meta(db_manager.get_recent_attacks(limit=50))
    stats = db_manager.get_statistics()
    known_attack_types = [row.get('attack_type') for row in attack_type_stats.get('by_type', [])]

    return render_template('attacks.html',
                         attack_type_stats=attack_type_stats,
                         recent_attacks=recent_attacks,
                         attack_meta=ATTACK_META,
                         attack_catalog=build_attack_catalog(known_attack_types),
                         stats=stats)


@app.route('/api/attacks/stats')
@login_required
def api_attack_stats():
    return jsonify(normalize_attack_type_stats(db_manager.get_attack_type_stats()))


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Upload CSV/JSON file and run prediction.

    The selected model_type controls whether CICIDS2017 or IIoT2025 XGBoost is used.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash(t('flash.file_not_selected'), 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash(t('flash.file_not_selected'), 'error')
            return redirect(request.url)

        if not (file and allowed_file(file.filename)):
            flash(t('flash.invalid_file_type'), 'error')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        model_type = (request.form.get('model_type') or 'cicids').strip().lower()

        if model_type not in SUPPORTED_UPLOAD_MODELS:
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(t('flash.file_read_error', error='Unsupported model_type. Choose cicids or iiot2025.'), 'error')
            return redirect(request.url)

        try:
            # Read uploaded file only to detect its schema.
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filepath, low_memory=False)
            else:
                df = pd.read_json(filepath)

            df.columns = df.columns.str.strip()

            if model_type == IIOT2025_MODEL_ID:
                iiot_results = predict_iiot2025_dataframe(df)
                predictions = []
                attack_type_counts = {}
                first_attack_prediction_id = None
                highest_attack_confidence = 0.0

                for idx, result in enumerate(iiot_results):
                    normalized = normalize_iiot2025_result(result, row=idx + 1)
                    attack_type_counts[normalized['attack_type']] = attack_type_counts.get(normalized['attack_type'], 0) + 1
                    highest_attack_confidence = max(highest_attack_confidence, normalized['confidence'])

                    prediction_id = save_cicids_prediction(normalized)
                    if first_attack_prediction_id is None:
                        first_attack_prediction_id = prediction_id

                    predictions.append({
                        'row': idx + 1,
                        'prediction': normalized['prediction'],
                        'raw_prediction': normalized['raw_prediction'],
                        'probability': normalized['attack_probability'],
                        'attack_probability': normalized['attack_probability'],
                        'confidence': normalized['confidence'],
                        'model_name': normalized['model_name'],
                        'attack_type': normalized['attack_type'],
                        'attack_type_confidence': normalized['attack_type_confidence'],
                        'attack_type_meta': ATTACK_META.get(normalized['attack_type'], ATTACK_META.get('Unknown', {})),
                    })

                attacks_total = len(predictions)
                benign_total = 0
                attack_distribution = []
                for attack_type, count in sorted(attack_type_counts.items(), key=lambda item: item[1], reverse=True):
                    percentage = round((count / attacks_total) * 100, 1) if attacks_total else 0.0
                    attack_distribution.append({
                        'type': attack_type,
                        'count': count,
                        'percentage': percentage,
                        'meta': ATTACK_META.get(attack_type, ATTACK_META.get('Unknown', {})),
                    })

                alert_result = None
                if attacks_total > 0:
                    alert_result = send_telegram_summary_alert(
                        filename=filename,
                        model_name=IIOT2025_MODEL_NAME,
                        total_rows=len(iiot_results),
                        attack_rows=attacks_total,
                        benign_rows=benign_total,
                        attack_distribution=attack_distribution,
                        highest_confidence=highest_attack_confidence,
                        lang=get_current_language(),
                    )
                    if first_attack_prediction_id and alert_result and alert_result.get('success'):
                        db_manager.save_alert(first_attack_prediction_id, alert_result.get('message', 'Telegram summary alert sent.'))

                summary = build_prediction_summary(predictions)
                limited_predictions = predictions[:500]
                os.remove(filepath)

                return render_template(
                    'upload_results.html',
                    predictions=limited_predictions,
                    total=len(iiot_results),
                    attacks=attacks_total,
                    benign=benign_total,
                    attack_distribution=attack_distribution,
                    alert_result=alert_result,
                    model_name=IIOT2025_MODEL_NAME,
                    filename=filename,
                    detected_classes=summary['detected_classes'],
                    class_counts=summary['class_counts'],
                    total_rows=summary['total_rows'],
                )

            is_cicids2017 = model_type == 'cicids'

            if is_cicids2017:
                # New CICIDS2017 ML model.
                cicids_results = predict_cicids_dataframe(df)

                predictions = []
                attack_type_counts = {}
                benign_saved = 0
                first_attack_prediction_id = None
                highest_attack_confidence = 0.0

                for idx, result in enumerate(cicids_results):
                    normalized = normalize_cicids_result(result, row=idx + 1)
                    is_attack = normalized['is_attack']

                    if is_attack:
                        attack_type_counts[normalized['attack_type']] = attack_type_counts.get(normalized['attack_type'], 0) + 1
                        highest_attack_confidence = max(highest_attack_confidence, normalized['confidence'])

                    # Keep every detected attack, but cap benign rows in history.
                    if is_attack or benign_saved < BENIGN_HISTORY_LIMIT:
                        prediction_id = save_cicids_prediction(normalized)
                        if is_attack and first_attack_prediction_id is None:
                            first_attack_prediction_id = prediction_id
                        if not is_attack:
                            benign_saved += 1

                    predictions.append({
                        'row': idx + 1,
                        'prediction': normalized['prediction'],
                        'raw_prediction': normalized['raw_prediction'],
                        'probability': normalized['attack_probability'],
                        'attack_probability': normalized['attack_probability'],
                        'confidence': normalized['confidence'],
                        'model_name': normalized['model_name'],
                        'attack_type': normalized['attack_type'],
                        'attack_type_confidence': normalized['attack_type_confidence'],
                        'attack_type_meta': ATTACK_META.get(normalized['attack_type'], ATTACK_META.get('Unknown', {})),
                    })

                attacks_total = sum(1 for p in predictions if p['prediction'] == 'Attack')
                benign_total = len(cicids_results) - attacks_total

                attack_distribution = []
                for attack_type, count in sorted(attack_type_counts.items(), key=lambda item: item[1], reverse=True):
                    percentage = round((count / attacks_total) * 100, 1) if attacks_total else 0.0
                    attack_distribution.append({
                        'type': attack_type,
                        'count': count,
                        'percentage': percentage,
                        'meta': ATTACK_META.get(attack_type, ATTACK_META.get('Unknown', {})),
                    })

                alert_result = None
                if attacks_total > 0:
                    alert_result = send_telegram_summary_alert(
                        filename=filename,
                        model_name=CICIDS_MODEL_NAME,
                        total_rows=len(cicids_results),
                        attack_rows=attacks_total,
                        benign_rows=benign_total,
                        attack_distribution=attack_distribution,
                        highest_confidence=highest_attack_confidence,
                        lang=get_current_language(),
                    )
                    if first_attack_prediction_id and alert_result and alert_result.get('success'):
                        db_manager.save_alert(first_attack_prediction_id, alert_result.get('message', 'Telegram summary alert sent.'))

                # Show only first 500 rows in browser. Full CICIDS CSV can have 200k+ rows.
                summary = build_prediction_summary(predictions)
                limited_predictions = predictions[:500]

                os.remove(filepath)

                return render_template(
                    'upload_results.html',
                    predictions=limited_predictions,
                    total=len(cicids_results),
                    attacks=attacks_total,
                    benign=benign_total,
                    attack_distribution=attack_distribution,
                    alert_result=alert_result,
                    model_name=CICIDS_MODEL_NAME,
                    filename=filename,
                    detected_classes=summary['detected_classes'],
                    class_counts=summary['class_counts'],
                    total_rows=summary['total_rows'],
                )

            # Legacy model for old NSL-KDD style CSV/JSON data.
            results = predict_batch(df)

            predictions = []
            first_attack_prediction_id = None
            highest_attack_confidence = 0.0
            for idx, result in enumerate(results):
                if result['success']:
                    attack_type_info = build_attack_type_info(result)
                    prediction_id = db_manager.save_prediction(
                        model_name=result['model_name'],
                        prediction=result['prediction'],
                        probability=result['probability'],
                        confidence=result['confidence'],
                        alert_sent=False,
                        attack_type=attack_type_info['type'] if attack_type_info else None,
                        attack_type_confidence=attack_type_info['confidence'] if attack_type_info else None,
                    )

                    if result['prediction'] == 'Attack':
                        highest_attack_confidence = max(highest_attack_confidence, float(result.get('confidence', 0.0) or 0.0))
                        if first_attack_prediction_id is None:
                            first_attack_prediction_id = prediction_id

                    predictions.append({
                        'row': idx + 1,
                        'prediction': result['prediction'],
                        'probability': result['probability'],
                        'confidence': result['confidence'],
                        'attack_type': attack_type_info['type'] if attack_type_info else 'None',
                        'attack_type_confidence': attack_type_info['confidence'] if attack_type_info else None,
                        'attack_type_meta': attack_type_info['meta'] if attack_type_info else {},
                    })

            attacks_total = sum(1 for p in predictions if p['prediction'] == 'Attack')
            attack_counts = {attack_type: 0 for attack_type in ['DoS', 'Probe', 'R2L', 'U2R']}
            for pred in predictions:
                if pred['prediction'] != 'Attack':
                    continue
                pred_type = pred.get('attack_type')
                if pred_type in attack_counts:
                    attack_counts[pred_type] += 1

            attack_distribution = []
            for attack_type, count in attack_counts.items():
                percentage = round((count / attacks_total) * 100, 1) if attacks_total else 0.0
                attack_distribution.append({
                    'type': attack_type,
                    'count': count,
                    'percentage': percentage,
                    'meta': ATTACK_META.get(attack_type, ATTACK_META['Unknown']),
                })

            alert_result = None
            if attacks_total > 0:
                alert_result = send_telegram_summary_alert(
                    filename=filename,
                    model_name='Legacy IDS fallback',
                    total_rows=len(results),
                    attack_rows=attacks_total,
                    benign_rows=len(results) - attacks_total,
                    attack_distribution=attack_distribution,
                    highest_confidence=highest_attack_confidence,
                )
                if first_attack_prediction_id and alert_result and alert_result.get('success'):
                    db_manager.save_alert(first_attack_prediction_id, alert_result.get('message', 'Telegram summary alert sent.'))

            os.remove(filepath)

            return render_template(
                'upload_results.html',
                predictions=predictions,
                total=len(results),
                attacks=attacks_total,
                benign=len(results) - attacks_total,
                attack_distribution=attack_distribution,
                alert_result=alert_result,
                model_name='Legacy IDS fallback',
            )

        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            flash(t('flash.file_read_error', error=str(e)), 'error')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/settings')
@login_required
def settings():
    bot_info = get_bot_info()
    models = get_cicids_models()
    
    return render_template('settings.html',
                         bot_info=bot_info,
                         models=models,
                         alert_threshold=os.getenv('ALERT_THRESHOLD', '0.7'))


@app.route('/model-plots/<path:filename>')
@login_required
def model_plot(filename):
    """Serve generated model plot images."""
    plot_dir = MODEL_DIR / 'plots'
    file_path = plot_dir / filename
    if not file_path.exists() or not file_path.is_file():
        abort(404)
    return send_from_directory(plot_dir, filename)


# API ROUTES

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """
    
    Request Body:
        {
            "features": {
                "duration": 0,
                "protocol_type": "tcp",
                ...
            },
            "model": "xgboost" (optional)
        }
    
    Response:
        {
            "success": true,
            "prediction": "Attack" | "Normal",
            "probability": 0.85,
            "confidence": 0.85,
            "model_name": "xgboost",
            "feature_importance": {...}
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'Missing data'}), 400
        
        model_type = (data.get('model_type') or data.get('model') or 'cicids').strip().lower()
        if model_type not in SUPPORTED_UPLOAD_MODELS:
            return jsonify({'success': False, 'error': 'Unsupported model_type. Choose cicids or iiot2025.'}), 400

        features = data.get('features', data)  # Accept both flat payload and nested features.
        feature_df = pd.DataFrame([features])

        if model_type == IIOT2025_MODEL_ID:
            raw_result = predict_iiot2025_dataframe(feature_df)[0]
            result = normalize_iiot2025_result(raw_result)
        else:
            raw_result = predict_cicids_dataframe(feature_df)[0]
            result = normalize_cicids_result(raw_result)
        
        if result['success']:
            attack_type_info = result.get('attack_type_info')
            prediction_id = None
            if result['is_attack']:
                prediction_id = save_cicids_prediction(result, features=features)
            
            if result['prediction'] == 'Attack' and result['attack_probability'] > 0.7:
                alert_result = send_telegram_alert(
                    prediction=result['prediction'],
                    probability=result['attack_probability'],
                    confidence=result['confidence'],
                    model_name=result['model_name'],
                    features=features,
                    attack_type_info=attack_type_info,
                )
                result['alert_sent'] = alert_result.get('success', False)
                if prediction_id and result['alert_sent']:
                    db_manager.save_alert(prediction_id, alert_result.get('message', 'Alert sent'))
            
            summary = build_prediction_summary([result])
            result.update({
                'model_name': result.get('model_name'),
                'detected_classes': summary['detected_classes'],
                'total_rows': summary['total_rows'],
                'class_counts': summary['class_counts'],
            })
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"API predict error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
@login_required
def api_predict_batch():
    """
    
    Request Body:
        {
            "data": [
                {"duration": 0, ...},
                {"duration": 1, ...}
            ],
            "model": "xgboost"
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'success': False, 'error': 'Missing data'}), 400

        model_type = (data.get('model_type') or data.get('model') or 'cicids').strip().lower()
        if model_type not in SUPPORTED_UPLOAD_MODELS:
            return jsonify({'success': False, 'error': 'Unsupported model_type. Choose cicids or iiot2025.'}), 400

        batch_data = data['data']
        if not isinstance(batch_data, list):
            return jsonify({'success': False, 'error': 'Missing data'}), 400

        batch_df = pd.DataFrame(batch_data)
        if model_type == IIOT2025_MODEL_ID:
            raw_results = predict_iiot2025_dataframe(batch_df)
            results = [normalize_iiot2025_result(result, row=idx + 1) for idx, result in enumerate(raw_results)]
            model_name = IIOT2025_MODEL_NAME
        else:
            raw_results = predict_cicids_dataframe(batch_df)
            results = [normalize_cicids_result(result, row=idx + 1) for idx, result in enumerate(raw_results)]
            model_name = CICIDS_MODEL_NAME

        benign_saved = 0
        for result in results:
            if result.get('is_attack') or benign_saved < BENIGN_HISTORY_LIMIT:
                save_cicids_prediction(result)
                if not result.get('is_attack'):
                    benign_saved += 1

        attack_type_counts = {}
        for result in results:
            if result.get('prediction') != 'Attack':
                continue
            attack_type = result.get('attack_type')
            attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1
        
        attacks_total = sum(1 for r in results if r.get('prediction') == 'Attack')
        summary = build_prediction_summary(results)
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results),
            'total_rows': summary['total_rows'],
            'attacks': attacks_total,
            'benign': len(results) - attacks_total,
            'model_name': model_name,
            'detected_classes': summary['detected_classes'],
            'class_counts': summary['class_counts'],
            'attack_type_stats': attack_type_counts,
        })
        
    except Exception as e:
        logger.error(f"API batch predict error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
@login_required
def api_history():
    limit = request.args.get('limit', 100, type=int)
    predictions = db_manager.get_prediction_history(limit=limit)
    
    return jsonify({
        'success': True,
        'predictions': predictions,
        'count': len(predictions)
    })


@app.route('/api/history/clear', methods=['POST'])
@login_required
def api_clear_history():
    try:
        result = db_manager.clear_all_history()
        logger.info(f"History cleared: {result['predictions_deleted']} predictions, {result['alerts_deleted']} alerts")
        return jsonify({
            'success': True,
            'message': f"History cleared. {result['predictions_deleted']} records deleted.",
            'deleted': result
        })
    except Exception as e:
        logger.error(f"History clear error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/statistics', methods=['GET'])
@login_required
def api_statistics():
    stats = db_manager.get_statistics()
    models = get_cicids_models()
    
    return jsonify({
        'success': True,
        'statistics': stats,
        'models': models
    })


@app.route('/api/models', methods=['GET'])
@login_required
def api_models():
    models = get_cicids_models()
    
    return jsonify({
        'success': True,
        'models': models
    })


@app.route('/api/models/<model_name>/metrics', methods=['GET'])
@login_required
def api_model_metrics(model_name):
    if model_name in (CICIDS_MODEL_ID, CICIDS_MODEL_NAME):
        return jsonify({
            'success': True,
            'model': get_cicids_models()[0],
            'message': 'No saved metrics are available for this model.'
        })

    if model_name in (IIOT2025_MODEL_ID, IIOT2025_MODEL_NAME):
        return jsonify({
            'success': True,
            'model': get_cicids_models()[1],
            'message': 'Metrics are available for this model.'
        })

    return jsonify({
        'success': False,
        'error': 'Model not found'
    }), 404


@app.route('/api/telegram/test', methods=['POST'])
@login_required
def api_test_telegram():
    try:
        data = request.get_json(silent=True) or {}

        threshold = data.get("threshold", 70)

        lang = get_current_language()

        texts = {
            "kk": f"""✅ IDS тест хабарламасы

Telegram боты дұрыс жұмыс істеп тұр.
Ескерту шегі: {threshold}%

Модель: CICIDS2017 XGBoost
Күйі: Белсенді""",

            "ru": f"""✅ Тестовое сообщение IDS

Telegram бот работает корректно.
Порог уведомления: {threshold}%

Модель: CICIDS2017 XGBoost
Статус: Активен""",

            "en": f"""✅ IDS Test Message

Telegram bot is working correctly.
Alert threshold: {threshold}%

Model: CICIDS2017 XGBoost
Status: Active"""
        }

        result = send_telegram_alert(
            prediction="Attack",
            probability=1.0,
            confidence=1.0,
            model_name="CICIDS2017 XGBoost",
            attack_type_info={
                "custom_message": texts.get(lang, texts["kk"])
            },
            lang=lang,
            force=True
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': bool(predictor and predictor.get('is_loaded')),
        'model_name': predictor.get('model_name') if predictor else CICIDS_MODEL_NAME,
    })


# UTILITIES

def get_sample_values():
    return get_cicids_sample_values()


# ERROR HANDLERS

@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': t('error.not_found')}), 404
    return render_template('error.html', error_code=404, error_message=t('error.not_found')), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': t('error.server')}), 500
    return render_template('error.html', error_code=500, error_message=t('error.server')), 500


# MAIN

if __name__ == '__main__':
    init_app()
    
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('PORT', 5001))
    
    logger.info(f"IDS server started: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

