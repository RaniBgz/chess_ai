        if is_training:
            try:
                train_game_number = 1
                move_none = ChessState.Move((0,0), (0,0), gs.board)
                move_data = move_queue.get_nowait()
                print("Move data: ", move_data)
                if move_data == 'error':
                    training_error = True
                    is_training = False
                    continue
                if move_data is None:
                    is_training = False
                    continue
                visualize, game_number, is_trained, move, game_ended = move_data
                # print(f'Train on game {game_number}')
                if game_number and game_number>train_game_number:
                    train_game_number = game_number
                # print("Is trained: ", is_trained)
                if visualize == "reset":
                    gs = ChessState.GameState()  # Reset the game state
                    validMoves = gs.getValidMoves()
                    continue
                if not is_trained:
                    if visualize:
                        move_player = move_none.fromChessNotation(move, gs.board)
                        for i in range(len(validMoves)):
                            if move_player == validMoves[i]:
                                gs.makeMove(validMoves[i])
                                moveMade = True
                                animate = True
                    if game_ended:
                        gs = ChessState.GameState()  # Reset the game state
                        validMoves = gs.getValidMoves()
                else:
                    print("Training Finished!")
                    training_process.terminate()
                    move_queue.close()
                    is_training = False
            except multiprocessing.queues.Empty:
                pass

