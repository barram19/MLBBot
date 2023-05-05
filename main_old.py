import requests
import numpy as np
import xgboost as xgb

# Set API endpoint URL and API key
url = 'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey=00ab27442da4a2b1f8460c5c70d0b3d8&bookmakers=fanduel&markets=h2h,spreads,totals&oddsFormat=american'
api_key = '00ab27442da4a2b1f8460c5c70d0b3d8'
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
              game_data.append([home_team, away_team, commence_time, home_odds, away_odds])

    # Convert the game data to a numpy array and split it into features and labels
    game_data = np.array(game_data)
    X = game_data[:, 3:].astype(float)
    y = np.where(game_data[:, 0] == game_data[:, 1], 2, np.where(game_data[:, 3] > game_data[:, 4], 1, 0))

    # Train the XGB model on the data
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # Make predictions on new data using the trained model
    new_data = X
    y_pred = model.predict(new_data)

    # Output the predicted labels along with the matchup and date
    for i, game in enumerate(game_data):
        home_team = game[0]
        away_team = game[1]
        date = game[2]
        prediction = y_pred[i]
        if prediction == 0:
            print(f"{date}: {away_team} vs {home_team}: {away_team} to win")
        elif prediction == 1:
            print(f"{date}: {away_team} vs {home_team}: {home_team} to win")
        else:
            print(f"{date}: {away_team} vs {home_team}: Draw")
else:
    print("Failed to retrieve odds data")
