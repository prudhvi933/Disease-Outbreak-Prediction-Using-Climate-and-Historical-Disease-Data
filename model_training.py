import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping
import joblib
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath, target_disease='Acute Diarrhoeal Disease'):
    """Loads data, filters for disease, cleans types, sorts, and imputes missing values."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='latin1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Filter for target disease
    print(f"Filtering for disease: {target_disease}")
    df = df[df['Disease'] == target_disease].copy()
    print(f"Retained {len(df)} records.")

    # Rename column
    df = df.rename(columns={'state_ut': 'stateut'})
    
    # Select columns
    cols = ['stateut', 'district', 'day', 'mon', 'year', 'Latitude', 'Longitude', 'preci', 'LAI', 'Temp', 'Cases']
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    df = df[cols]

    # Type conversion
    df['Cases'] = pd.to_numeric(df['Cases'], errors='coerce')
    df = df.dropna(subset=['Cases'])

    # Sort by time
    df = df.sort_values(by=['stateut', 'district', 'year', 'mon', 'day'])

    # Handle numeric columns
    numeric_cols = ['Temp', 'preci', 'LAI']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Impute missing climate data
    print("Imputing missing climate data...")
    for col in numeric_cols:
        df[col] = df[col].fillna(df.groupby(['stateut', 'district', 'mon'])[col].transform('mean'))
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    return df


def create_features(df):
    """Creates lag features."""
    print("Creating lag features...")
    df['caseslastweek'] = df.groupby(['stateut', 'district'])['Cases'].shift(1)
    df['caseslastmonth'] = df.groupby(['stateut', 'district'])['Cases'].shift(4)
    df = df.dropna(subset=['caseslastweek', 'caseslastmonth'])
    return df


def encode_and_scale(df):
    """Encodes categorical variables and scales features."""
    print("Encoding and scaling...")
    
    le_state = LabelEncoder()
    df['stateut_enc'] = le_state.fit_transform(df['stateut'])
    
    le_district = LabelEncoder()
    df['district_enc'] = le_district.fit_transform(df['district'])
    
    scaler = StandardScaler()
    feature_cols = ['Temp', 'preci', 'LAI']
    df[[f+'_scaled' for f in feature_cols]] = scaler.fit_transform(df[feature_cols])
    
    return df, scaler, le_state, le_district


def build_lstm(input_shape):
    """Builds LSTM model."""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def prepare_lstm_data(X, y, sequence_length=4):
    """Prepares sequential data for LSTM."""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    # Load and preprocess
    df = load_and_preprocess_data('Final_data.csv')
    if df is not None:
        df = create_features(df)
        df, scaler, le_state, le_district = encode_and_scale(df)
        
        # Define features
        feature_cols = ['day', 'mon', 'year', 'Latitude', 'Longitude', 
                        'Temp_scaled', 'preci_scaled', 'LAI_scaled', 
                        'caseslastweek', 'caseslastmonth', 'stateut_enc', 'district_enc']
        
        # Prepare data
        X = df[feature_cols].values
        y = df['Cases'].values
        
        # Create sequences for LSTM
        print("\nPreparing LSTM sequences...")
        seq_len = 4
        X_seq, y_seq = prepare_lstm_data(X, y, seq_len)
        
        # Time-aware split (80/20)
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Build and train LSTM
        print("\nTraining LSTM...")
        lstm_model = build_lstm((seq_len, X.shape[1]))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        lstm_model.fit(X_train, y_train, 
                       validation_data=(X_test, y_test),
                       epochs=20, batch_size=32, callbacks=[early_stop], verbose=1)
        
        # Evaluate LSTM
        y_pred = lstm_model.predict(X_test, verbose=0)
        lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        lstm_mae = mean_absolute_error(y_test, y_pred)
        lstm_mape = mean_absolute_percentage_error(y_test, y_pred)
        
        print(f"\n--- LSTM Metrics ---")
        print(f"RMSE: {lstm_rmse:.2f}")
        print(f"MAE: {lstm_mae:.2f}")
        print(f"MAPE: {lstm_mape:.2f}")
        
        print(f"\nBest Model selected: LSTM")
        
        # Save pipeline
        pipeline = {
            'scaler': scaler,
            'le_state': le_state,
            'le_district': le_district,
            'model': lstm_model,
            'model_type': 'LSTM',
            'features': feature_cols,
            'sequence_length': seq_len
        }
        
        print("Saving LSTM pipeline...")
        joblib.dump(pipeline, 'best_disease_model.pkl')
        lstm_model.save('best_disease_model.keras')
        
        print("Done! LSTM model saved successfully.")
