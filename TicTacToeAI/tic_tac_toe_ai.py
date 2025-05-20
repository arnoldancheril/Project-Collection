import math
import tkinter as tk
from tkinter import messagebox
import random

def check_winner(board):
    # Check rows, columns, and diagonals for a winner
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != ' ':
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != ' ':
            return board[0][i]
    
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return board[0][2]
    
    return None

def is_moves_left(board):
    for row in board:
        if ' ' in row:
            return True
    return False

def minimax(board, depth, is_max):
    score = check_winner(board)
    
    if score == 'X':
        return -10 + depth
    if score == 'O':
        return 10 - depth
    if not is_moves_left(board):
        return 0
    
    if is_max:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    best = max(best, minimax(board, depth + 1, False))
                    board[i][j] = ' '
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    best = min(best, minimax(board, depth + 1, True))
                    board[i][j] = ' '
        return best

def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val
    
    return best_move

def on_click(row, col):
    global player_turn, board, buttons, player_symbol, ai_symbol, player_score, ai_score
    if board[row][col] == ' ' and player_turn:
        board[row][col] = player_symbol
        buttons[row][col].config(text=player_symbol)
        player_turn = False
        update_turn_label()
        winner = check_winner(board)
        if winner == player_symbol:
            player_score += 1
            update_score_label()
            messagebox.showinfo("Game Over", "You win!")
            ask_reset()
        elif not is_moves_left(board):
            messagebox.showinfo("Game Over", "It's a draw!")
            ask_reset()
        else:
            ai_move()

def ai_move():
    global player_turn, board, buttons, ai_symbol, player_score, ai_score
    row, col = find_best_move(board)
    board[row][col] = ai_symbol
    buttons[row][col].config(text=ai_symbol)
    player_turn = True
    update_turn_label()
    winner = check_winner(board)
    if winner == ai_symbol:
        ai_score += 1
        update_score_label()
        messagebox.showinfo("Game Over", "AI wins!")
        ask_reset()
    elif not is_moves_left(board):
        messagebox.showinfo("Game Over", "It's a draw!")
        ask_reset()

def reset_game():
    global board, buttons, player_turn
    for i in range(3):
        for j in range(3):
            board[i][j] = ' '
            buttons[i][j].config(text=' ')
    player_turn = (player_symbol == 'X')
    update_turn_label()
    if not player_turn:
        ai_move()

def ask_reset():
    if messagebox.askyesno("Reset Game", "Would you like to play again?"):
        reset_game()
    else:
        root.quit()

def choose_symbol():
    global player_symbol, ai_symbol, player_turn
    choose_window = tk.Toplevel(root)
    choose_window.title("Choose Symbol")
    tk.Label(choose_window, text="Choose your symbol:", font=('normal', 15)).pack()
    
    def set_symbol(symbol):
        global player_symbol, ai_symbol, player_turn
        player_symbol = symbol
        ai_symbol = 'O' if symbol == 'X' else 'X'
        player_turn = (player_symbol == 'X')
        update_turn_label()
        choose_window.destroy()
        if not player_turn:
            ai_move()
    
    tk.Button(choose_window, text="X", font=('normal', 15), command=lambda: set_symbol('X')).pack(pady=5)
    tk.Button(choose_window, text="O", font=('normal', 15), command=lambda: set_symbol('O')).pack(pady=5)

def update_score_label():
    score_label.config(text=f"Player: {player_score}  AI: {ai_score}")

def update_turn_label():
    if player_turn:
        turn_label.config(text="Player's turn")
    else:
        turn_label.config(text="AI's turn")

def main():
    global root, buttons, player_turn, board, player_symbol, ai_symbol, player_score, ai_score, score_label, turn_label
    root = tk.Tk()
    root.title("Tic-Tac-Toe")
    
    board = [[' ' for _ in range(3)] for _ in range(3)]
    buttons = [[None for _ in range(3)] for _ in range(3)]
    player_score = 0
    ai_score = 0
    
    score_label = tk.Label(root, text=f"Player: {player_score}  AI: {ai_score}", font=('normal', 20))
    score_label.grid(row=4, column=0, columnspan=3)
    
    turn_label = tk.Label(root, text="", font=('normal', 15))
    turn_label.grid(row=5, column=0, columnspan=3)
    
    for i in range(3):
        for j in range(3):
            buttons[i][j] = tk.Button(root, text=' ', font=('normal', 40), width=5, height=2,
                                     command=lambda row=i, col=j: on_click(row, col))
            buttons[i][j].grid(row=i, column=j)
    
    reset_button = tk.Button(root, text='Reset', font=('normal', 20), command=reset_game)
    reset_button.grid(row=3, column=0, columnspan=3)
    
    choose_symbol()
    
    root.mainloop()

if __name__ == "__main__":
    main()