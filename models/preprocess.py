"""
Data Preprocessing for IDS
Желі трафигін өңдеу және дайындау
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
from pathlib import Path

# Модельдер сақталатын жол
MODEL_DIR = Path(__file__).parent / 'saved_models'
MODEL_DIR.mkdir(exist_ok=True)

# Негізгі feature аттары (KDD Cup 1999 форматына негізделген)
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Сандық және категориялық бағандар
NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']


class DataPreprocessor:
    """
    Желі трафигін өңдеу класы
    Орнықты (robust) тәсілдерді қолдану
    """
    
    def __init__(self, use_robust=True):
        self.use_robust = use_robust
        self.numeric_scaler = RobustScaler() if use_robust else StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.fitted = False
        
    def fit(self, X):
        """
        Препроцессорды үйрету
        
        Args:
            X: DataFrame немесе numpy array
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=FEATURE_NAMES[:X.shape[1]])
        
        # Сандық бағандарды өңдеу
        numeric_cols = [col for col in NUMERIC_FEATURES if col in X.columns]
        if numeric_cols:
            # Missing values өңдеу
            X_numeric = X[numeric_cols].replace('', np.nan).replace('?', np.nan)
            X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
            
            # Imputer үйрету
            self.imputer.fit(X_numeric)
            X_imputed = self.imputer.transform(X_numeric)
            
            # Scaler үйрету
            self.numeric_scaler.fit(X_imputed)
        
        # Категориялық бағандарды өңдеу
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                le = LabelEncoder()
                # Жаңа мәндер үшін резерв
                values = X[col].fillna('unknown').astype(str).unique().tolist() + ['unknown']
                le.fit(values)
                self.label_encoders[col] = le
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """
        Деректерді өңдеу
        
        Args:
            X: DataFrame немесе numpy array
        
        Returns:
            Өңделген numpy array
        """
        if not self.fitted:
            raise ValueError("Препроцессор үйретілмеген. Алдымен fit() шақырыңыз.")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=FEATURE_NAMES[:X.shape[1]])
        
        result_dfs = []
        
        # Сандық бағандарды өңдеу
        numeric_cols = [col for col in NUMERIC_FEATURES if col in X.columns]
        if numeric_cols:
            X_numeric = X[numeric_cols].replace('', np.nan).replace('?', np.nan)
            X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce')
            
            # Missing values өңдеу
            X_imputed = self.imputer.transform(X_numeric)
            
            # Масштабтау
            X_scaled = self.numeric_scaler.transform(X_imputed)
            
            result_dfs.append(pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index))
        
        # Категориялық бағандарды өңдеу
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                le = self.label_encoders.get(col)
                if le:
                    # Жаңа мәндерді 'unknown' деп белгілеу
                    values = X[col].fillna('unknown').astype(str)
                    values = values.apply(lambda x: x if x in le.classes_ else 'unknown')
                    encoded = le.transform(values)
                    result_dfs.append(pd.DataFrame({col: encoded}, index=X.index))
        
        # Барлық бағандарды біріктіру
        if result_dfs:
            X_processed = pd.concat(result_dfs, axis=1)
            return X_processed.values
        
        return X.values
    
    def fit_transform(self, X):
        """Fit және transform бірге"""
        return self.fit(X).transform(X)
    
    def save(self, filepath):
        """Препроцессорды сақтау"""
        joblib.dump({
            'numeric_scaler': self.numeric_scaler,
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'fitted': self.fitted,
            'use_robust': self.use_robust
        }, filepath)
    
    def load(self, filepath):
        """Препроцессорды жүктеу"""
        data = joblib.load(filepath)
        self.numeric_scaler = data['numeric_scaler']
        self.imputer = data['imputer']
        if not hasattr(self.imputer, '_fill_dtype') and hasattr(self.imputer, 'statistics_'):
            self.imputer._fill_dtype = self.imputer.statistics_.dtype
        self.label_encoders = data['label_encoders']
        self.fitted = data['fitted']
        self.use_robust = data['use_robust']
        return self


def validate_input_data(data, expected_features=41):
    """
    Енгізілген деректерді тексеру
    
    Args:
        data: dict, list, немесе DataFrame
        expected_features: Күтілетін feature саны
    
    Returns:
        dict: {'valid': bool, 'message': str, 'data': processed_data}
    """
    try:
        # DataFrame-ге түрлендіру
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=FEATURE_NAMES[:data.shape[1]])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            return {
                'valid': False,
                'message': f'Қолдау көрсетілмейтін деректер форматы: {type(data)}',
                'data': None
            }
        
        # Бос деректерді тексеру
        if df.empty:
            return {
                'valid': False,
                'message': 'Деректер бос',
                'data': None
            }
        
        # Required feature schema by name (order-independent input support)
        required_features = FEATURE_NAMES[:expected_features]
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 'unknown' if feature in CATEGORICAL_FEATURES else 0
        
        # Сандық бағандарды тексеру
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Категориялық бағандарды тексеру
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = 'unknown'
        
        return {
            'valid': True,
            'message': 'Деректер сәтті тексерілді',
            'data': df
        }
    
    except Exception as e:
        return {
            'valid': False,
            'message': f'Деректерді тексеру қатесі: {str(e)}',
            'data': None
        }


def create_preprocessing_pipeline(use_robust=True):
    """
    sklearn Pipeline құру
    
    Args:
        use_robust: RobustScaler қолдану
    
    Returns:
        sklearn Pipeline
    """
    scaler = RobustScaler() if use_robust else StandardScaler()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', scaler)
    ])
    
    # Категориялық transformer (егер қажет болса)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor


def detect_outliers(X, contamination=0.1):
    """
    Аномалияларды анықтау (Outlier detection)
    
    Args:
        X: Деректер
        contamination: Аномалиялардың болжамды үлесі
    
    Returns:
        outlier_mask: True - аномалия емес, False - аномалия
    """
    from sklearn.ensemble import IsolationForest
    
    clf = IsolationForest(contamination=contamination, random_state=42)
    y_pred = clf.fit_predict(X)
    
    return y_pred == 1  # 1 - нормалды, -1 - аномалия


def add_noise_robustness(X, noise_factor=0.01):
    """
    Шуға төзімділікті тексеру үшін шу қосу
    
    Args:
        X: Бастапқы деректер
        noise_factor: Шу деңгейі
    
    Returns:
        Шу қосылған деректер
    """
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise
