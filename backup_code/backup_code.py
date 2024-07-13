        # if is_training:
        #     try:
        #         train_game_number = 1
        #         move_none = ChessState.Move((0,0), (0,0), gs.board)
        #         move_data = move_queue.get_nowait()
        #         print("Move data: ", move_data)
        #         if move_data == 'error':
        #             training_error = True
        #             is_training = False
        #             continue
        #         if move_data is None:
        #             is_training = False
        #             continue
        #         visualize, game_number, is_trained, move, game_ended = move_data
        #         # print(f'Train on game {game_number}')
        #         if game_number and game_number>train_game_number:
        #             train_game_number = game_number
        #         # print("Is trained: ", is_trained)
        #         if visualize == "reset":
        #             gs = ChessState.GameState()  # Reset the game state
        #             validMoves = gs.getValidMoves()
        #             continue
        #         if not is_trained:
        #             if visualize:
        #                 move_player = move_none.fromChessNotation(move, gs.board)
        #                 for i in range(len(validMoves)):
        #                     if move_player == validMoves[i]:
        #                         gs.makeMove(validMoves[i])
        #                         moveMade = True
        #                         animate = True
        #             if game_ended:
        #                 gs = ChessState.GameState()  # Reset the game state
        #                 validMoves = gs.getValidMoves()
        #         else:
        #             print("Training Finished!")
        #             training_process.terminate()
        #             move_queue.close()
        #             is_training = False
        #     except multiprocessing.queues.Empty:
        #         pass
        #
        #
        # def train_ai(ai, queue):
        #     try:
        #         game_number_last = 1
        #         move_count = 0
        #         visualize = False
        #         moves_train = ai.train_on_pgn(PGN_PATH, num_games=NUM_GAMES_TRAIN)
        #         for game_number, is_trained, move, game_ended in moves_train:
        #             # print(f'Train on game {game_number}')
        #             move_count += 1
        #             if game_number == game_number_last + 1:
        #                 move_count = 0
        #                 game_number_last = game_number
        #                 queue.put(("reset", None, None, None, True))  # Signal to reset the board
        #
        #             if move_count <= 10:
        #                 visualize = True
        #             else:
        #                 visualize = False
        #             queue.put((visualize, game_number, is_trained, move, game_ended))
        #             if is_trained == True:
        #                 ai.save_model(config['model_path'])
        #                 save_config(config)
        #                 break
        #         queue.put(None)  # Signal the end of training
        #     except FileNotFoundError:
        #         print("ERROR: Training file not found or inaccessible")
        #         queue.put('error')
        #         queue.put(None)
        #     except Exception as e:
        #         print(f"ERROR: {e}")
        #         queue.put('error')
        #         queue.put(None)


'''
Old function to get best moves
'''

        # move_scores = []
        # for move in legal_moves:
        #     start_square_score = start_predictions[move.from_square]
        #     end_square_score = end_predictions[move.to_square]
        #     move_score = start_square_score * end_square_score
        #     move_scores.append((move, move_score))
        #     # print (f"Move: {move}, Score: {move_score}")