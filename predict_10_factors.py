import zipfile
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# 1. Load Data
def load_data():
    transfers_zip = 'data/transfers.zip'
    ratings_zip = 'data/ratings.zip'
    
    with zipfile.ZipFile(transfers_zip) as z:
        with z.open('top250-00-19.csv') as f:
            transfers = pd.read_csv(f)
            
    fifa_data = []
    with zipfile.ZipFile(ratings_zip) as z:
        # We'll use 2015-2019 to match the transfer seasons
        for year in [15, 16, 17, 18, 19]:
            fname = f'players_{year}.csv'
            if fname in z.namelist():
                with z.open(fname) as f:
                    df = pd.read_csv(f)
                    df['fifa_year'] = 2000 + year
                    fifa_data.append(df)
    
    fifa = pd.concat(fifa_data, ignore_index=True)
    return transfers, fifa

# 2. Preprocessing & Feature Engineering
def parse_fifa_stat(stat):
    if isinstance(stat, str):
        # Handle formats like '94+3' or '94-1'
        res = re.split(r'[+-]', stat)
        return float(res[0])
    return float(stat)

def engineer_features(transfers, fifa):
    # Clean transfers
    transfers = transfers.dropna(subset=['Transfer_fee', 'Season', 'Name']).copy()
    transfers['Season_Year'] = transfers['Season'].apply(lambda x: int(x.split('-')[0]))
    
    # Filter transfers to 2015-2018 (where we have FIFA data)
    transfers = transfers[transfers['Season_Year'].isin([2014, 2015, 2016, 2017, 2018])].copy()
    
    # Financial strength of buying club: 3-year rolling mean of the buying league's median transfer fee
    annual_league_medians = transfers.groupby(['League_to', 'Season_Year'])['Transfer_fee'].median().reset_index()
    annual_league_medians = annual_league_medians.sort_values(['League_to', 'Season_Year'])
    annual_league_medians['Buying_League_Strength'] = annual_league_medians.groupby('League_to')['Transfer_fee'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    transfers = transfers.merge(annual_league_medians[['League_to', 'Season_Year', 'Buying_League_Strength']], on=['League_to', 'Season_Year'], how='left')
    
    # Fuzzy match players (simplified for this task)
    # In a real scenario, we'd do a more robust join. 
    # Here we'll do a simple name + year join to keep it fast, or small sample fuzzy.
    
    # Prepare FIFA data: average stats across years for the same player-year if needed, 
    # but usually we want the rating BEFORE the transfer.
    # Transfer in 2015-2016 (Season_Year 2015) -> use FIFA 15 or 16.
    # We'll match on Name and Season_Year
    
    fifa['Name_Match'] = fifa['short_name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()
    transfers['Name_Match'] = transfers['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()
    
    # Map FIFA years to Season years
    # FIFA 15 is released in late 2014, used for 2014-2015 season.
    fifa['Season_Match'] = fifa['fifa_year'] - 1 # FIFA 15 -> Season 2014
    
    merged = transfers.merge(fifa, left_on=['Name_Match', 'Season_Year'], right_on=['Name_Match', 'Season_Match'], how='inner')
    
    # Define 10 Factors
    # 1. Contract duration (years left)
    def calc_duration(row):
        try:
            val = row['contract_valid_until']
            if pd.isna(val): return 2.0 # default
            return float(val) - row['Season_Year']
        except: return 2.0
    
    merged['Contract_Duration'] = merged.apply(calc_duration, axis=1)
    
    # 2. Age
    merged['Age_Feature'] = merged['age']
    
    # 3. Financial strength of buying club
    merged['Financial_Strength'] = merged['Buying_League_Strength']
    
    # 4. Ability (Overall)
    merged['Ability_Overall'] = merged['overall']
    
    # 5. Potential
    merged['Ability_Potential'] = merged['potential']
    
    # 6. Advanced Stats: xG Proxy (Finishing + Positioning)
    merged['xG_Proxy'] = merged['attacking_finishing'].apply(parse_fifa_stat) + merged['mentality_positioning'].apply(parse_fifa_stat)
    
    # 7. Advanced Stats: xA Proxy (Vision + Crossing)
    merged['xA_Proxy'] = merged['mentality_vision'].apply(parse_fifa_stat) + merged['attacking_crossing'].apply(parse_fifa_stat)
    
    # 8. Passport Premium (Binary: Top Nations)
    top_nations = ['Brazil', 'Argentina', 'France', 'Germany', 'Spain', 'England', 'Italy', 'Portugal', 'Netherlands', 'Belgium']
    merged['Passport_Premium'] = merged['nationality'].apply(lambda x: 1 if x in top_nations else 0)
    
    # 9. Position (Simplified categories)
    def group_pos(pos):
        if pos in ['ST', 'CF', 'LW', 'RW', 'LS', 'RS', 'RF', 'LF']: return 3 # Forward
        if pos in ['CAM', 'CM', 'CDM', 'LM', 'RM', 'LAM', 'RAM', 'LDM', 'RDM']: return 2 # Midfield
        if pos in ['CB', 'LB', 'RB', 'LWB', 'RWB', 'LCB', 'RCB']: return 1 # Defense
        return 0 # GK or other
    
    merged['Position_Feature'] = merged['player_positions'].apply(lambda x: group_pos(x.split(',')[0]))
    
    # 10. Home Nation Transfer (Replacing International Reputation)
    league_to_country = {
        'Premier League': 'England', ' England': 'England', 'League One': 'England', 'Championship': 'England',
        'LaLiga': 'Spain', 'LaLiga2': 'Spain', 'Primera División': 'Spain',
        'Serie A': 'Italy', 'Serie B': 'Italy', 'Serie C - B': 'Italy',
        '1.Bundesliga': 'Germany', 'Bundesliga': 'Germany', '2.Bundesliga': 'Germany',
        'Ligue 1': 'France', 'Ligue 2': 'France',
        'Liga NOS': 'Portugal', ' Portugal': 'Portugal', 'Ledman Liga Pro': 'Portugal',
        'Eredivisie': 'Netherlands',
        'Série A': 'Brazil', ' Brazil': 'Brazil',
        'Süper Lig': 'Turkey',
        'Premier Liga': 'Russia', ' Russia': 'Russia',
        'Jupiler Pro League': 'Belgium', ' Belgium': 'Belgium',
        'Super League': 'China', ' China': 'China',
        'MLS': 'United States',
        'Argentina': 'Argentina', 'Torneo Final': 'Argentina',
        'Mexico': 'Mexico', 'Liga MX Clausura': 'Mexico', 'Liga MX Apertura': 'Mexico',
        'Scotland': 'Scotland', 'Premiership': 'Scotland'
    }
    
    def is_home_transfer(row):
        dest_country = league_to_country.get(row['League_to'], 'Unknown')
        return 1 if row['nationality'] == dest_country else 0

    merged['Home_Nation_Transfer'] = merged.apply(is_home_transfer, axis=1)
    
    features = [
        'Contract_Duration', 'Age_Feature', 'Financial_Strength', 
        'Ability_Overall', 'Ability_Potential', 'xG_Proxy', 'xA_Proxy', 
        'Passport_Premium', 'Position_Feature', 'Home_Nation_Transfer'
    ]
    
    X = merged[features]
    y = merged['Transfer_fee']
    
    return X, y, features, merged

# 3. Model & Explainability
def run_model(X, y, features, df_full):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("--- Evaluation Metrics ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.savefig('shap_summary.png')
    plt.close()
    
    # --- LIME Explanations for Archetypes ---
    explainer_lime = LimeTabularExplainer(
        X_train.values, 
        feature_names=features, 
        class_names=['Transfer_fee'], 
        mode='regression',
        random_state=42
    )
    
    # Archetype 1: High Overall/Potential (Superstar)
    superstar = df_full[df_full['overall'] > 85].head(1)
    if not superstar.empty:
        idx = superstar.index[0]
        # Map back to X_test index or just use the row from df_full
        row = df_full.loc[idx][features].values.astype(float)
        exp = explainer_lime.explain_instance(row, model.predict)
        print(f"\n--- Domain LIME: Superstar archetype ({df_full.loc[idx]['Name']}) ---")
        for feature, weight in exp.as_list():
            print(f"{feature}: {weight:.4f}")

    # Archetype 2: Young Talent (U21, High Potential)
    young_talent = df_full[(df_full['age'] < 22) & (df_full['potential'] > 80)].head(1)
    if not young_talent.empty:
        idx = young_talent.index[0]
        row = df_full.loc[idx][features].values.astype(float)
        exp = explainer_lime.explain_instance(row, model.predict)
        print(f"\n--- Domain LIME: Young Talent archetype ({df_full.loc[idx]['Name']}) ---")
        for feature, weight in exp.as_list():
            print(f"{feature}: {weight:.4f}")

    # Archetype 3: Veteran
    veteran = df_full[df_full['age'] > 30].head(1)
    if not veteran.empty:
        idx = veteran.index[0]
        row = df_full.loc[idx][features].values.astype(float)
        exp = explainer_lime.explain_instance(row, model.predict)
        print(f"\n--- Domain LIME: Veteran archetype ({df_full.loc[idx]['Name']}) ---")
        for feature, weight in exp.as_list():
            print(f"{feature}: {weight:.4f}")

    print("\n--- Feature Definitions ---")
    print("1. Contract_Duration: Years remaining on contract at time of transfer.")
    print("2. Age_Feature: Player age in years.")
    print("3. Financial_Strength: 3-year rolling mean of the median transfer fee of the BUYING league.")
    print("4. Ability_Overall: FIFA Overall rating (quantifies current technical/physical standing).")
    print("5. Ability_Potential: FIFA Potential rating (quantifies future ceiling).")
    print("6. xG_Proxy (Advanced Stat): Sum of FIFA 'Finishing' and 'Positioning' stats.")
    print("7. xA_Proxy (Advanced Stat): Sum of FIFA 'Vision' and 'Crossing' stats.")
    print("8. Passport_Premium: Binary (1 if player is from a top 10 FIFA nation, 0 otherwise).")
    print("9. Position_Feature: Categorical (0:GK, 1:DEF, 2:MID, 3:FWD).")
    print("10. Home_Nation_Transfer: Binary (1 if player is transferring to a club in their home country, 0 otherwise).")

    return model

if __name__ == "__main__":
    print("Loading data...")
    transfers, fifa = load_data()
    print("Engineering features...")
    X, y, features, df_full = engineer_features(transfers, fifa)
    print(f"Dataset size: {len(X)} rows")
    if len(X) > 0:
        run_model(X, y, features, df_full)
    else:
        print("No matching data found. Check matching logic.")
