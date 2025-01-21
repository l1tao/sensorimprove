import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def load_data(boxmod_path, juul_path):
    '''
      firt colomn: ground truth
      the rest colomns: sensor readings
      return: boxmod_X, boxmod_y, juul_X, juul_y
    '''
    boxmod_df = pd.read_csv(boxmod_path, header=None)
    juul_df = pd.read_csv(juul_path, header=None)
    
    # BoxMod (known environment)
    boxmod_y = boxmod_df.iloc[:, 0].values.reshape(-1, 1)
    boxmod_X = boxmod_df.iloc[:, 1:].values
    
    # JUUL (unknown environment)
    juul_y = juul_df.iloc[:, 0].values.reshape(-1, 1)
    juul_X = juul_df.iloc[:, 1:].values
    
    return boxmod_X, boxmod_y, juul_X, juul_y

def evaluate_model(true_values, predictions):
    """
        return: R^2 RMSE
    """
    r2 = r2_score(true_values, predictions)
    rmse = sqrt(mean_squared_error(true_values, predictions))
    return r2, rmse

def do_kfold_cv(X, y, model, X_scaler, y_scaler, n_splits=5):
    """
        KFold n_splits cross validation
        X_scaler, y_scaler: standardscaler and inverse_transform
        return: (mean_r2, mean_rmse, std_r2, std_rmse)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    cv_r2 = []
    cv_rmse = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # train model
        model.fit(X_train, y_train.ravel())
        
        # validation set predict after scaler
        y_val_pred_scaled = model.predict(X_val)
        
        # inverse_transform to original scale
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1,1)).ravel()
        y_val_true = y_scaler.inverse_transform(y_val)
        
        r2, rmse = evaluate_model(y_val_true, y_val_pred)
        cv_r2.append(r2)
        cv_rmse.append(rmse)
    
    return (np.mean(cv_r2), np.mean(cv_rmse), np.std(cv_r2), np.std(cv_rmse))

if __name__ == "__main__":
    # load data
    boxmod_path = 'e:/ML_challenges/data/BoxMod.csv'
    juul_path = 'e:/ML_challenges/data/JUUL.csv'
    boxmod_X, boxmod_y, juul_X, juul_y = load_data(boxmod_path, juul_path)
    
    # standardize
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # BoxMod fit scaler
    boxmod_X_scaled = X_scaler.fit_transform(boxmod_X)
    boxmod_y_scaled = y_scaler.fit_transform(boxmod_y)
    
    # transform JUUL
    juul_X_scaled = X_scaler.transform(juul_X)
    juul_y_scaled = y_scaler.transform(juul_y)

    '''LinearRegression
    '''
    lr_model= LinearRegression()

    mean_r2, mean_rmse, std_r2, std_rmse = do_kfold_cv(
        boxmod_X_scaled,
        boxmod_y_scaled,
        lr_model,
        X_scaler,
        y_scaler,
        n_splits=5
    )
    print("=== Linear Regression 5-Fold CV (BoxMod) ===")
    print(f"R2: {mean_r2:.4f} ± {std_r2:.4f}, RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    lr_model.fit(boxmod_X_scaled, boxmod_y_scaled.ravel())
    juul_pred_scaled = lr_model.predict(juul_X_scaled)
    juul_pred = y_scaler.inverse_transform(juul_pred_scaled.reshape(-1,1)).ravel()
    juul_true = y_scaler.inverse_transform(juul_y_scaled)
    r2_test, rmse_test = evaluate_model(juul_true, juul_pred)
    print("=== Linear Regression Test (JUUL) ===")
    print(f"R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}\n")
      
    
    ''' RandomForestRegressor
    '''
    rf_model = RandomForestRegressor(
        n_estimators=100,       
        max_depth=None,         
        random_state=1,
        n_jobs=-1               
    )
    
    mean_r2, mean_rmse, std_r2, std_rmse = do_kfold_cv(
        boxmod_X_scaled, 
        boxmod_y_scaled, 
        rf_model, 
        X_scaler, 
        y_scaler, 
        n_splits=5
    )
    print("=== Random Forest 5-Fold CV (BoxMod) ===")
    print(f"R2: {mean_r2:.4f} ± {std_r2:.4f}, RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    rf_model.fit(boxmod_X_scaled, boxmod_y_scaled.ravel())
    juul_pred_scaled = rf_model.predict(juul_X_scaled)
    juul_pred = y_scaler.inverse_transform(juul_pred_scaled.reshape(-1,1)).ravel()
    juul_true = y_scaler.inverse_transform(juul_y_scaled)
    r2_test, rmse_test = evaluate_model(juul_true, juul_pred)
    print("=== Random Forest Test (JUUL) ===")
    print(f"R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}\n")
    
    '''GradientBoostingRegressor
    '''
    gbr_model = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=1
    )
    
    mean_r2, mean_rmse, std_r2, std_rmse = do_kfold_cv(
        boxmod_X_scaled, 
        boxmod_y_scaled, 
        gbr_model, 
        X_scaler, 
        y_scaler, 
        n_splits=5
    )
    print("=== Gradient Boosting 5-Fold CV (BoxMod) ===")
    print(f"R2: {mean_r2:.4f} ± {std_r2:.4f}, RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    gbr_model.fit(boxmod_X_scaled, boxmod_y_scaled.ravel())
    juul_pred_scaled = gbr_model.predict(juul_X_scaled)
    juul_pred = y_scaler.inverse_transform(juul_pred_scaled.reshape(-1,1)).ravel()
    r2_test, rmse_test = evaluate_model(juul_true, juul_pred)
    print("=== Gradient Boosting Test (JUUL) ===")
    print(f"R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}\n")
    
    '''MLPRegressor
    '''
    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 64), 
        activation='relu', 
        solver='adam', 
        max_iter=200, 
        random_state=1
    )
    
    mean_r2, mean_rmse, std_r2, std_rmse = do_kfold_cv(
        boxmod_X_scaled, 
        boxmod_y_scaled, 
        nn_model, 
        X_scaler, 
        y_scaler, 
        n_splits=5
    )
    print("=== Neural Network (MLP) 5-Fold CV (BoxMod) ===")
    print(f"R2: {mean_r2:.4f} ± {std_r2:.4f}, RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    nn_model.fit(boxmod_X_scaled, boxmod_y_scaled.ravel())
    juul_pred_scaled = nn_model.predict(juul_X_scaled)
    juul_pred = y_scaler.inverse_transform(juul_pred_scaled.reshape(-1,1)).ravel()
    r2_test, rmse_test = evaluate_model(juul_true, juul_pred)
    print("=== Neural Network (MLP) Test (JUUL) ===")
    print(f"R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}")