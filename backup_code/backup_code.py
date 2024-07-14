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

'''
Old AI move logic
'''

        # for move in validMoves:
        #     if move.startRow == ai_move.from_square // 8 and move.startCol == ai_move.from_square % 8 and \
        #        move.endRow == ai_move.to_square // 8 and move.endCol == ai_move.to_square % 8:
        #         print("AI making move")
        #         cn_move = move.getChessNotation()
        #         print("Chess notation AI move: ", cn_move)
        #         metrics.score_move(gs, cn_move, n_top_moves=n_top_moves)
        #         gs.makeMove(move)
        #         moveMade = True
        #         animate = True
        #         ai_move_made = True
        #         total_ai_moves += 1
        #         break
        # if not ai_move_made:
        #     # print("AI couldn't make a valid move. Choosing a random move.")
        #     import random
        #     if validMoves:
        #         random_move = random.choice(validMoves)
        #         cn_random_move = random_move.getChessNotation()
        #         print("Chess notation RANDOM MOVE REPLACED : ", cn_random_move)
        #         metrics.score_move(gs, cn_random_move, n_top_moves=n_top_moves)
        #         gs.makeMove(random_move)
        #         moveMade = True
        #         animate = True
        #         replaced_moves += 1
        #         total_ai_moves += 1
        #     else:
        #         print("No valid moves available. Game over.")
        #         gameOver = True


        # Normalize things: always use the same move notation = chess notation, to string.
        # def build_tree(self, gs, base_move=None):
        #     print("In build tree")
        #     current_depth = 0
        #     base_evaluation = self.evaluate_board(gs) #Evaluate current board position, before the AI plays
        #
        #     self.root = Node(move=base_move,
        #                      evaluation=base_evaluation,
        #                      depth=0.0,
        #                      parent=None)
        #     self._build_tree_recursive(gs, self.root, 0)
        #     # for child in self.root.children:
        #     #     print(f"Child: {child}")


        # def _build_tree_recursive(self, gs, current_node, current_depth, alpha, beta, maximizing_player):
        #     if current_depth >= self.max_depth:
        #         return
        #
        #     top_moves = self.ai.get_top_n_moves(gs, self.width)
        #
        #     for move in top_moves:
        #         move_obj = Move.fromChessNotation(move, gs.board)
        #         gs.makeMove(move_obj)
        #         move_evaluation = self.evaluate_board(gs)
        #         print("Move evaluation: ", move_evaluation)
        #         print("Alpha: ", alpha)
        #         print("Beta: ", beta)
        #         child_node = Node(move=move, evaluation=move_evaluation, depth=current_depth + 0.5, parent=current_node)
        #         current_node.add_child(child_node)
        #
        #         if maximizing_player:
        #             alpha = max(alpha, move_evaluation)
        #             if alpha >= beta:
        #                 gs.undoMove()
        #                 print("Pruning in maximizing player")
        #                 break
        #         else:
        #             beta = min(beta, move_evaluation)
        #             if beta <= alpha:
        #                 gs.undoMove()
        #                 print("Pruning in minimizing player")
        #                 break
        #
        #         self._build_tree_recursive(gs, child_node, current_depth + 0.5, alpha, beta, not maximizing_player)
        #         gs.undoMove()
