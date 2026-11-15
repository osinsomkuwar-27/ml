
import random
import pickle

# 1. Tic-Tac-Toe Environment
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9

    def display(self):
        print("\n       TIC TAC TOE")
        print("   +-----+-----+-----+")
        for i in range(0, 9, 3):
            print("   |  " + "  |  ".join(self.board[i:i+3]) + "  |")
            print("   +-----+-----+-----+")
        print()

    def moves(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, pos, player):
        if pos in self.moves():
            self.board[pos] = player
            return True
        return False

    def winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8),
                (0,3,6), (1,4,7), (2,5,8),
                (0,4,8), (2,4,6)]

        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]
        return None


# 2. Q-Learning Agent
class QLearningAgent:
    def __init__(self):
        self.q = {}
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

    def get_state(self, board, player):
        if player == 'X':
            return tuple(board)
        return tuple('O' if x == 'X' else 'X' if x == 'O' else ' '
                     for x in board)

    def choose_action(self, board, player):
        state = self.get_state(board, player)
        moves = [i for i in range(9) if board[i] == ' ']

        if random.random() < self.epsilon:
            return random.choice(moves)

        values = [self.q.get((state, a), 0) for a in moves]
        best = max(values)
        choices = [moves[i] for i in range(len(moves))
                   if values[i] == best]

        return random.choice(choices)

    def update(self, state, action, reward, next_state):
        old = self.q.get((state, action), 0)
        future = 0

        if next_state is not None:
            moves = [i for i in range(9) if next_state[i] == ' ']
            if moves:
                future = max(self.q.get((next_state, a), 0)
                             for a in moves)

        self.q[(state, action)] = old + self.lr * (
            reward + self.gamma * future - old
        )


# 3. Training using Self-Play
agent = QLearningAgent()

for episode in range(10000):
    game = TicTacToe()
    history = {'X': [], 'O': []}
    player = 'X'

    while game.moves() and not game.winner():
        state = agent.get_state(game.board, player)
        action = agent.choose_action(game.board, player)

        history[player].append((state, action))
        game.make_move(action, player)

        player = 'O' if player == 'X' else 'X'

    result = game.winner()

    for p in ['X', 'O']:
        reward = 1 if result == p else -1 if result else 0
        next_state = None

        for state, action in reversed(history[p]):
            agent.update(state, action, reward, next_state)
            next_state = state
            reward = 0

    agent.epsilon = max(0.01, agent.epsilon * 0.9995)

print("\nTraining Completed!")


# 4. Save Trained Model
with open("tictactoe.pkl", "wb") as f:
    pickle.dump(agent.q, f)

print("Model Saved Successfully!")


# 5. Testing Against Random Opponent
agent.epsilon = 0
wins = draws = losses = 0

for _ in range(1000):
    game = TicTacToe()

    while game.moves() and not game.winner():
        action = agent.choose_action(game.board, 'X')
        game.make_move(action, 'X')

        if game.winner() or not game.moves():
            break

        action = random.choice(game.moves())
        game.make_move(action, 'O')

    if game.winner() == 'X':
        wins += 1
    elif game.winner() == 'O':
        losses += 1
    else:
        draws += 1

print("\nTesting Results (1000 Games)")
print("Wins:", wins)
print("Losses:", losses)
print("Draws:", draws)
print("Win Rate:", round(wins / 1000 * 100, 2), "%")


# 6. Play Against Human
game = TicTacToe()

print("\nYou are X and Computer is O")
game.display()

while game.moves() and not game.winner():

    try:
        pos = int(input("Enter position (0-8): "))
    except ValueError:
        print("Enter a valid number!")
        continue

    if not game.make_move(pos, 'X'):
        print("Invalid move! Try again.")
        continue

    game.display()

    if game.winner() or not game.moves():
        break

    action = agent.choose_action(game.board, 'O')
    game.make_move(action, 'O')

    print("Computer chose:", action)
    game.display()

if game.winner() == 'X':
    print("Congratulations! You Win!")
elif game.winner() == 'O':
    print("Computer Wins!")
else:
    print("Match Draw!")