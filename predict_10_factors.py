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
    
    # Map FIFA years to Season years (FIFA 15 released late 2014)
    fifa['Season_Match'] = fifa['fifa_year'] - 1
    
    fifa['Short_Name_Match'] = fifa['short_name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()
    fifa['Long_Name_Match'] = fifa['long_name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()
    transfers['Name_Match'] = transfers['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower()

    # Join on both Short Name and Long Name to maximize matching accuracy
    merged_short = transfers.merge(fifa, left_on=['Name_Match', 'Season_Year'], right_on=['Short_Name_Match', 'Season_Match'], how='inner')
    merged_long = transfers.merge(fifa, left_on=['Name_Match', 'Season_Year'], right_on=['Long_Name_Match', 'Season_Match'], how='inner')
    
    merged = pd.concat([merged_short, merged_long]).drop_duplicates(subset=['Name', 'Season_Year']).reset_index(drop=True)
    
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
    
    # Feature Ranges Explanation
    print("\n--- Feature Ranges & Context ---")
    stats = X.describe()
    for feat in features:
        f_min = stats.loc['min', feat]
        f_max = stats.loc['max', feat]
        f_mean = stats.loc['mean', feat]
        print(f"{feat:22}: Range [{f_min:6.1f} - {f_max:6.1f}] | Mean: {f_mean:6.1f}")

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
    plt.savefig('plots/shap_summary.png')
    plt.close()
    
    # --- LIME Explanations for Archetypes ---
    explainer_lime = LimeTabularExplainer(
        X_train.values, 
        feature_names=features, 
        class_names=['Transfer_fee'], 
        mode='regression',
        random_state=42
    )
    
    scenario_specs = [
        {
            'label': 'Young Brazilian Talent',
            'player_name': 'Richarlison',
            'season': 2017,
            'desc': 'Young prospect moving from Brazil to the Premier League (Fluminense to Watford).'
        },
        {
            'label': 'English Domestic Move',
            'player_name': 'Alex Oxlade-Chamberlain',
            'season': 2017,
            'desc': 'English player moving domestically between top clubs (Arsenal to Liverpool).'
        },
        {
            'label': 'Superstar Juggernaut',
            'player_name': 'Paul Pogba',
            'season': 2016,
            'desc': 'Marquee signing with world-record fee context (Juve to Man Utd).'
        },
        {
            'label': 'Veteran Superstar',
            'player_name': 'Cristiano Ronaldo',
            'season': 2018,
            'desc': 'Elite veteran (33) moving for a high fee to a top league (Real to Juve).'
        },
        {
            'label': 'Mid-tier Competitive',
            'player_name': 'Daley Blind',
            'season': 2018,
            'desc': 'Prime-age established player moving between competitive leagues (Man Utd to Ajax).'
        }
    ]

    print("\n--- Feature Definitions & Expected Ranges ---")
    print("1. Contract_Duration: [0 - 7] Years left on contract. Mean: 3.5. High (>5) = Security premium.")
    print("2. Age_Feature: [17 - 35] Player age. Mean: 25. High (>28) typically discounts fee.")
    print("3. Financial_Strength: [€5M - €15M] 3-yr rolling median fee of buying league. High = Premier League context.")
    print("4. Ability_Overall: [50 - 94] Current FIFA rating. Elite (>85) exponentially increases value.")
    print("5. Ability_Potential: [60 - 94] FIFA Potential. High (>85) adds 'future' premium.")
    print("6. xG_Proxy (Advanced): [20 - 190] Sum of Finishing/Positioning. Elite strikers sit > 160.")
    print("7. xA_Proxy (Advanced): [40 - 170] Sum of Vision/Crossing. Playmakers sit > 140.")
    print("8. Passport_Premium: [0 or 1] 1 for top 10 FIFA nations (e.g., Brazil, France, England).")
    print("9. Position_Feature: [0 - 3] 0:GK, 1:DEF, 2:MID, 3:FWD. Forwards typically carry higher fees.")
    print("10. Home_Nation_Transfer: [0 or 1] 1 if transferring within home country league system.")

    for spec in scenario_specs:
        # Use boolean indexing for better reliability than query()
        mask = (df_full['Name'] == spec['player_name']) & (df_full['Season_Year'] == spec['season'])
        candidates = df_full[mask]
        
        if candidates.empty:
            # Fallback if specific player not found (though they should be)
            candidates = df_full.head(1)
        
        # Pick the first matching row and its data
        target_row = candidates.iloc[0]
        row_values = target_row[features].values.astype(float)
        
        # LIME Explanation
        exp = explainer_lime.explain_instance(row_values, model.predict, num_features=10)
        
        # Nice Plot Styling
        lime_df = pd.DataFrame(exp.as_list(), columns=['feature', 'weight'])
        lime_df = lime_df.sort_values('weight')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2a9d8f' if val > 0 else '#e76f51' for val in lime_df['weight']]
        ax.barh(lime_df['feature'], lime_df['weight'], color=colors, alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        
        pred_val = model.predict(row_values.reshape(1, -1))[0]
        actual_val = target_row['Transfer_fee']
        
        ax.set_title(f"{spec['label']} | Pred €{pred_val/1e6:.2f}M | Actual €{actual_val/1e6:.2f}M", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('LIME local contribution (Weight)')
        ax.grid(axis='x', alpha=0.3)
        
        # Metadata box
        player_name = target_row['Name']
        nation = target_row['nationality']
        age = target_row['age']
        metadata = f"{spec['desc']}\nPlayer: {player_name}\nNation: {nation}\nAge: {age}"
        ax.text(0.98, 0.05, metadata, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
        
        plt.tight_layout()
        fname = f"lime_{spec['label'].lower().replace(' ', '_')}.png"
        plt.savefig(f"plots/{fname}")
        plt.close()
        print(f"Generated plot: {fname} for {player_name}")

    return model

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
