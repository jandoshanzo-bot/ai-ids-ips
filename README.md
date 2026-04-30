<<<<<<< HEAD
# 🛡️ IDS - Intrusion Detection System

Статистикалық және орнықты машиналық оқыту әдістеріне негізделген желілік шабуылдарды анықтау жүйесі.

## 📋 Мазмұны

- [Жоба туралы](#жоба-туралы)
- [Мүмкіндіктер](#мүмкіндіктер)
- [Орнату](#орнату)
- [Қолдану](#қолдану)
- [API құжаттамасы](#api-құжаттамасы)
- [Жоба құрылымы](#жоба-құрылымы)
- [Telegram баптау](#telegram-баптау)

## 🎯 Жоба туралы

Бұл веб-қосымша желілік трафикті талдап, шабуылдарды анықтау үшін машиналық оқыту алгоритмдерін қолданады. Қосымша келесі модельдерді қолдайды:

- **Logistic Regression** - Статистикалық модель
- **Random Forest** - Ensemble модель

Шабуыл анықталған жағдайда Telegram-ға автоматты хабарлама жіберіледі.

## ✨ Мүмкіндіктер

### 🤖 Машиналық оқыту
- ✅ Logistic Regression және Random Forest модельдері
- ✅ RobustScaler қолдану (outlier-ге төзімді)
- ✅ Feature importance талдау
- ✅ Ensemble болжамдар

### 📊 Деректер енгізу
- ✅ Form арқылы қолмен енгізу
- ✅ CSV файл жүктеу
- ✅ JSON API арқылы деректер қабылдау

### 📱 Хабарландырулар
- ✅ Telegram Bot интеграциясы
- ✅ Автоматты алерттер (threshold > 70%)
- ✅ Бөлек алерт модуль

### 📈 Статистикалар
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ Confusion Matrix
- ✅ Болжам тарихы
- ✅ SQLite дерекқоры

## 🚀 Орнату

### 1. Құпиялылықты клондау

```bash
git clone <repository-url>
cd ids_project
```

### 2. Орта айнымалыларын баптау

`.env` файлын өзгерту:

```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Alert Threshold
ALERT_THRESHOLD=0.7
```

### 3. Қажетті пакеттерді орнату

```bash
pip install -r requirements.txt
```

### 4. Қосымшаны іске қосу

```bash
python app.py
```

Қосымша `http://localhost:5001` адресінде қолжетімді болады.

## 💻 Қолдану

### Веб интерфейс

1. **Басты бет** - Жалпы статистика және модельдер ақпараты
2. **Болжам** - Жеке деректерді енгізіп талдау
3. **Жүктеу** - CSV/JSON файлдарды жүктеу
4. **Тарих** - Барлық болжамдар тарихы
5. **Статистика** - Толық метрикалар және графиктер
6. **Параметрлер** - Баптаулар

### API қолдану

#### Жеке болжам

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "duration": 0,
      "src_bytes": 215,
      "dst_bytes": 45076,
      "count": 5
    },
    "model": "random_forest"
  }'
```

#### Бірнеше болжам

```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"duration": 0, "src_bytes": 215},
      {"duration": 1, "src_bytes": 500}
    ],
    "model": "random_forest"
  }'
```

## 📚 API құжаттамасы

### Маршруттар

| Метод | Маршрут | Сипаттама |
|-------|---------|-----------|
| GET | `/` | Басты бет |
| GET | `/predict` | Болжам формасы |
| POST | `/predict` | Болжам жасау |
| GET | `/history` | Тарих беті |
| GET | `/statistics` | Статистика беті |
| GET | `/upload` | Файл жүктеу беті |
| POST | `/api/predict` | API: Жеке болжам |
| POST | `/api/predict/batch` | API: Бірнеше болжам |
| GET | `/api/history` | API: Тарих |
| GET | `/api/statistics` | API: Статистикалар |
| GET | `/api/models` | API: Модельдер тізімі |
| GET | `/api/health` | API: Денсаулық тексеру |

## 📁 Жоба құрылымы

```
ids_project/
│
├── app.py                 # Flask негізгі қосымшасы
├── requirements.txt       # Python пакеттері
├── .env                   # Орта айнымалылары
├── README.md              # Осы файл
│
├── models/                # ML модельдері
│   ├── __init__.py
│   ├── train_models.py    # Модельдерді оқыту
│   ├── predict.py         # Болжау функциялары
│   ├── preprocess.py      # Деректерді өңдеу
│   └── saved_models/      # Сақталған модельдер
│
├── utils/                 # Қосымша утилиттер
│   ├── __init__.py
│   ├── telegram_alert.py  # Telegram боты
│   └── database.py        # SQLite жұмысы
│
├── templates/             # HTML шаблондар
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── history.html
│   ├── statistics.html
│   ├── upload.html
│   ├── upload_results.html
│   ├── settings.html
│   └── error.html
│
├── static/                # CSS, JS файлдар
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── uploads/               # Жүктелген файлдар
└── data/                  # Дерекқор және деректер
```

## 🤖 Telegram баптау

### 1. BotFather-ден бот жасау

1. Telegram-да `@BotFather` табыңыз
2. `/newbot` командасын жіберіңіз
3. Ботқа ат және username беріңіз
4. **Bot Token**-ді сақтап алыңыз

### 2. Chat ID алу

1. Ботқа хабарлама жіберіңіз
2. Браузерде ашыңыз:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. `chat.id` мәнің табыңыз

### 3. .env файлын жаңарту

```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxyz
TELEGRAM_CHAT_ID=123456789
```

## 🔧 Қосымша баптаулар

### Модельдерді қайта оқыту

```python
from models.train_models import train_and_save_models

# Жаңа модельдерді оқыту
train_and_save_models(force_retrain=True)
```

### Alert threshold өзгерту

`.env` файлында:

```env
ALERT_THRESHOLD=0.8  # 80% үстінде болса алерт жіберу
```

## 📊 Метрикалар

Модельдер бағаланатын метрикалар:

- **Accuracy** - Дәлдік
- **Precision** - Нақтылық
- **Recall** - Толықтық
- **F1-Score** - F1 өлшемі
- **ROC-AUC** - ROC кривизны ауданы
- **Confusion Matrix** - Қателесу матрицасы

## 🛡️ Қауіпсіздік

- `.env` файлын ешқашан Git-ке жібермеңіз
- `SECRET_KEY` өндірісте өзгертіңіз
- `FLASK_DEBUG=False` өндірісте орнатыңыз

## 📝 Лицензия

MIT License

## 👨‍💻 Автор

Jandos

---

**Ескерту:** Бұл жоба демо мақсатында жасалған. Өндірісте қолданар алдында қосымша тестілеу қажет.
=======
# ai-ids-ips

