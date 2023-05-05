import requests
import numpy as np
import xgboost as xgb


# Set API endpoint URL and API key
url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey=8f283a0b9aa2f6e2b517c77b786bb1cb&bookmakers=fanduel&markets=h2h,spreads,totals&oddsFormat=american'
api_key = '8f283a0b9aa2f6e2b517c77b786bb1cb'
# Set headers for the request
headers = {
    'Content-Type': 'application/json',
    'x-api-key': api_key,
}

# Make the request and get the response data
response = requests.get(url, headers=headers)

if response.status_code == 200:
    odds_data = response.json()
    # Process the odds data

    # Extract the necessary information from the odds data
game_data = []
for game in odds_data:
    home_team = game['home_team']
    away_team = game['away_team']
    commence_time = game['commence_time'][:10] # Extract the date from the commence_time field
    bookmakers = game['bookmakers']
    for bookmaker in bookmakers:
        markets = bookmaker['markets']
        for market in markets:
            if market['key'] == 'h2h':
                outcomes = market['outcomes']
                home_odds = outcomes[0]['price']
                away_odds = outcomes[1]['price']
            elif market['key'] == 'totals':
                outcomes = market['outcomes']
                point = outcomes[0]['point']
                over_odds = outcomes[0]['price']
                under_odds = outcomes[1]['price']
                if point is not None:
                    game_data.append([home_team, away_team, commence_time, home_odds, away_odds, point, over_odds, under_odds])

# Convert the game data to a numpy array and split it into features and labels
game_data = np.array(game_data)
X = game_data[:, 3:7].astype(float)
y_win = np.where(game_data[:, 0] == game_data[:, 1], 2, np.where(game_data[:, 3] > game_data[:, 4], 1, 0))
y_total = np.where(game_data[:, 5].astype(float) > 8.5, 1, 0)

# Train the XGB model on the data for game winner
model_win = xgb.XGBClassifier()
model_win.fit(X, y_win)

# Train the XGB model on the data for over/under
model_total = xgb.XGBClassifier()
model_total.fit(X, y_total)

# Make predictions on new data using the trained model
new_data = X
y_pred_win = model_win.predict(new_data)
y_pred_total = model_total.predict(new_data)

# Output the predicted labels along with the matchup and date
for i, game in enumerate(game_data):
    home_team = game[0]
    away_team = game[1]
    date = game[2]
    win_prediction = y_pred_win[i]
    total_prediction = y_pred_total[i]
    if win_prediction == 0:
        print(f"{date}: {away_team} vs {home_team}: {away_team} ML")
    elif win_prediction == 1:
        print(f"{date}: {away_team} vs {home_team}: {home_team} ML")
    else:
        print(f"{date}: {away_team} vs {home_team}: Draw")
    if total_prediction == 0:
        print(f"u{game[5]}")
    else:
        print(f"o{game[5]}")
else:
    print("Failed to retrieve odds data")
