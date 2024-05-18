from django.shortcuts import render, redirect
import random
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Initialize moves
moves = ['rock', 'paper', 'scissors']

# Initialize data storage (this would typically be a database)
data = {
    'player_move': [],
    'opponent_move': [],
}

# Label encoder
le = LabelEncoder()
le.fit(moves)

# Function to determine outcome
def determine_outcome(player, opponent):
    if player == opponent:
        return 'draw'
    elif (player == 'rock' and opponent == 'scissors') or \
         (player == 'paper' and opponent == 'rock') or \
         (player == 'scissors' and opponent == 'paper'):
        return 'win'
    else:
        return 'lose'

# Function to get the model's move
def get_model_move(model, last_player_move):
    if not data['player_move']:
        return random.choice(moves)
    else:
        features = pd.DataFrame({'player_move_encoded': [le.transform([last_player_move])[0]]})
        return model.predict(features)[0]

# Index view
def index(request):
    return render(request, 'game/index.html')

# Play view
def play(request):
    if request.method == 'POST':
        player_move = request.POST.get('move')
        opponent_move = random.choice(moves)  # Default to random if no data
        model_move = opponent_move
        
        # Update data
        data['player_move'].append(player_move)
        data['opponent_move'].append(opponent_move)
        
        if len(data['player_move']) > 5:  # Start training after collecting some data
            df = pd.DataFrame(data)
            df['player_move_encoded'] = le.transform(df['player_move'])
            X = df[['player_move_encoded']]
            y = df['opponent_move']
            
            # Train the model
            model = DecisionTreeClassifier()
            model.fit(X, y)
            
            # Get the model's move
            model_move = get_model_move(model, player_move)
        
        outcome = determine_outcome(player_move, model_move)
        
        context = {
            'player_move': player_move,
            'opponent_move': model_move,
            'outcome': outcome,
        }
        return render(request, 'game/result.html', context)
    return redirect('index')
