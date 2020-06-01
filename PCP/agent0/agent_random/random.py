def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    x = []
    for i, element in enumerate(board[5,:]):
        if element == 0:
            x.append(i)
    action = np.random.choice(x)
    return action, saved_state