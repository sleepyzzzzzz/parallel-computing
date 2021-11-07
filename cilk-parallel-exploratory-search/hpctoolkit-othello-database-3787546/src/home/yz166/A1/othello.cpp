#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cilk/cilk.h>
#include <cilk/reducer_max.h>
#include <cilk/reducer_min.h>

#define BIT 0x1

#define X_BLACK 0
#define O_WHITE 1
#define OTHERCOLOR(c) (1-(c))

/* 
	represent game board squares as a 64-bit unsigned integer.
	these macros index from a row,column position on the board
	to a position and bit in a game board bitvector
*/
#define BOARD_BIT_INDEX(row,col) ((8 - (row)) * 8 + (8 - (col)))
#define BOARD_BIT(row,col) (0x1LL << BOARD_BIT_INDEX(row,col))
#define MOVE_TO_BOARD_BIT(m) BOARD_BIT(m.row, m.col)

/* all of the bits in the row 8 */
#define ROW8 \
  (BOARD_BIT(8,1) | BOARD_BIT(8,2) | BOARD_BIT(8,3) | BOARD_BIT(8,4) |	\
   BOARD_BIT(8,5) | BOARD_BIT(8,6) | BOARD_BIT(8,7) | BOARD_BIT(8,8))
			  
/* all of the bits in column 8 */
#define COL8 \
  (BOARD_BIT(1,8) | BOARD_BIT(2,8) | BOARD_BIT(3,8) | BOARD_BIT(4,8) |	\
   BOARD_BIT(5,8) | BOARD_BIT(6,8) | BOARD_BIT(7,8) | BOARD_BIT(8,8))

/* all of the bits in column 1 */
#define COL1 (COL8 << 7)

#define IS_MOVE_OFF_BOARD(m) (m.row < 1 || m.row > 8 || m.col < 1 || m.col > 8)
#define IS_DIAGONAL_MOVE(m) (m.row != 0 && m.col != 0)
#define MOVE_OFFSET_TO_BIT_OFFSET(m) (m.row * 8 + m.col)

typedef unsigned long long ull;

/* 
	game board represented as a pair of bit vectors: 
	- one for x_black disks on the board
	- one for o_white disks on the board
*/
typedef struct { ull disks[2]; } Board;

typedef struct { int row; int col; } Move;

// move and score struct used in minimax algorithm
typedef struct { int score; Move pos; } Minimax_Move;

// cilk reducer_max comparator
struct max_cmp {
  bool operator()(const Minimax_Move& x, const Minimax_Move& y) const { return x.score < y.score; }
};

// cilk reducer_min comparator
struct min_cmp {
  bool operator()(const Minimax_Move& x, const Minimax_Move& y) const { return x.score > y.score; }
};

Board start = { 
	BOARD_BIT(4,5) | BOARD_BIT(5,4) /* X_BLACK */, 
	BOARD_BIT(4,4) | BOARD_BIT(5,5) /* O_WHITE */
};
 
Move offsets[] = {
  {0,1}		/* right */,		{0,-1}		/* left */, 
  {-1,0}	/* up */,		{1,0}		/* down */, 
  {-1,-1}	/* up-left */,		{-1,1}		/* up-right */, 
  {1,1}		/* down-right */,	{1,-1}		/* down-left */
};

int noffsets = sizeof(offsets)/sizeof(Move);
char diskcolor[] = { '.', 'X', 'O', 'I' };


void PrintDisk(int x_black, int o_white)
{
  printf(" %c", diskcolor[x_black + (o_white << 1)]);
}

void PrintBoardRow(int x_black, int o_white, int disks)
{
  if (disks > 1) {
    PrintBoardRow(x_black >> 1, o_white >> 1, disks - 1);
  }
  PrintDisk(x_black & BIT, o_white & BIT);
}

void PrintBoardRows(ull x_black, ull o_white, int rowsleft)
{
  if (rowsleft > 1) {
    PrintBoardRows(x_black >> 8, o_white >> 8, rowsleft - 1);
  }
  printf("%d", rowsleft);
  PrintBoardRow((int)(x_black & ROW8),  (int) (o_white & ROW8), 8);
  printf("\n");
}

void PrintBoard(Board b)
{
  printf("  1 2 3 4 5 6 7 8\n");
  PrintBoardRows(b.disks[X_BLACK], b.disks[O_WHITE], 8);
}

/* 
	place a disk of color at the position specified by m.row and m,col,
	flipping the opponents disk there (if any) 
*/
void PlaceOrFlip(Move m, Board *b, int color) 
{
  ull bit = MOVE_TO_BOARD_BIT(m);
  b->disks[color] |= bit;
  b->disks[OTHERCOLOR(color)] &= ~bit;
}

/* 
	try to flip disks along a direction specified by a move offset.
	the return code is 0 if no flips were done.
	the return value is 1 + the number of flips otherwise.
*/
int TryFlips(Move m, Move offset, Board *b, int color, int verbose, int domove)
{
  Move next;
  next.row = m.row + offset.row;
  next.col = m.col + offset.col;

  if (!IS_MOVE_OFF_BOARD(next)) {
    ull nextbit = MOVE_TO_BOARD_BIT(next);
    if (nextbit & b->disks[OTHERCOLOR(color)]) {
      int nflips = TryFlips(next, offset, b, color, verbose, domove);
      if (nflips) {
      	if (verbose) printf("flipping disk at %d,%d\n", next.row, next.col);
      	if (domove) PlaceOrFlip(next,b,color);
      	return nflips + 1;
      }
    } else if (nextbit & b->disks[color]) return 1;
  }
  return 0;
} 

int FlipDisks(Move m, Board *b, int color, int verbose, int domove)
{
  int i;
  int nflips = 0;
	
  /* try flipping disks along each of the 8 directions */
  for(i=0;i<noffsets;i++) {
    int flipresult = TryFlips(m,offsets[i], b, color, verbose, domove);
    nflips += (flipresult > 0) ? flipresult - 1 : 0;
  }
  return nflips;
}

void ReadMove(int color, Board *b)
{
  Move m;
  ull movebit;
  for(;;) {
    printf("Enter %c's move as 'row,col': ", diskcolor[color+1]);
    scanf("%d,%d",&m.row,&m.col);
		
    /* if move is not on the board, move again */
    if (IS_MOVE_OFF_BOARD(m)) {
      printf("Illegal move: row and column must both be between 1 and 8\n");
      PrintBoard(*b);
      continue;
    }
    movebit = MOVE_TO_BOARD_BIT(m);
		
    /* if board position occupied, move again */
    if (movebit & (b->disks[X_BLACK] | b->disks[O_WHITE])) {
      printf("Illegal move: board position already occupied.\n");
      PrintBoard(*b);
      continue;
    }
		
    /* if no disks have been flipped */ 
    {
      int nflips = FlipDisks(m, b,color, 1, 1);
      if (nflips == 0) {
      	printf("Illegal move: no disks flipped\n");
      	PrintBoard(*b);
      	continue;
      }
      PlaceOrFlip(m, b, color);
      printf("You flipped %d disks\n", nflips);
      PrintBoard(*b);
    }
    break;
  }
}

/*
	return the set of board positions adjacent to an opponent's
	disk that are empty. these represent a candidate set of 
	positions for a move by color.
*/
Board NeighborMoves(Board b, int color)
{
  int i;
  Board neighbors = {0,0};
  for (i = 0;i < noffsets; i++) {
    ull colmask = (offsets[i].col != 0) ? 
      ((offsets[i].col > 0) ? COL1 : COL8) : 0;
    int offset = MOVE_OFFSET_TO_BIT_OFFSET(offsets[i]);

    if (offset > 0) {
      neighbors.disks[color] |= 
	(b.disks[OTHERCOLOR(color)] >> offset) & ~colmask;
    } else {
      neighbors.disks[color] |= 
	(b.disks[OTHERCOLOR(color)] << -offset) & ~colmask;
    }
  }
  neighbors.disks[color] &= ~(b.disks[X_BLACK] | b.disks[O_WHITE]);
  return neighbors;
}

/*
	return the set of board positions that represent legal
	moves for color. this is the set of empty board positions  
	that are adjacent to an opponent's disk where placing a
	disk of color will cause one or more of the opponent's
	disks to be flipped.
*/
int EnumerateLegalMoves(Board b, int color, Board *legal_moves)
{
  static Board no_legal_moves = {0,0};
  Board neighbors = NeighborMoves(b, color);
  ull my_neighbor_moves = neighbors.disks[color];
  int row;
  int col;
	
  int num_moves = 0;
  *legal_moves = no_legal_moves;
	
  for(row=8; row >=1; row--) {
    ull thisrow = my_neighbor_moves & ROW8;
    for(col=8; thisrow && (col >= 1); col--) {
      if (thisrow & COL8) {
      	Move m = { row, col };
      	if (FlipDisks(m, &b, color, 0, 0) > 0) {
      	  legal_moves->disks[color] |= BOARD_BIT(row,col);
      	  num_moves++;
      	}
      }
      thisrow >>= 1;
    }
    my_neighbor_moves >>= 8;
  }
  return num_moves;
}

int HumanTurn(Board *b, int color)
{
  Board legal_moves;
  int num_moves = EnumerateLegalMoves(*b, color, &legal_moves);
  if (num_moves > 0) {
    ReadMove(color, b);
    return 1;
  } else return 0;
}

int CountBitsOnBoard(Board *b, int color)
{
  ull bits = b->disks[color];
  int ndisks = 0;
  for (; bits ; ndisks++) {
    bits &= bits - 1; // clear the least significant bit set
  }
  return ndisks;
}

/* 
  return a vector of all legal moves for color. Modified from EnumerateLegalMoves function, 
  this is the set of empty board positions that are adjacent to an opponent's disk where placing a
  disk of color will cause one or more of the opponent's disks to be flipped.
*/
std::vector<Move> find_all_legal_moves(Board b, int color) {
  // an empty vector that will be returned by the function
  std::vector<Move> legal_moves;
  Board neighbors = NeighborMoves(b, color);
  ull my_neighbor_moves = neighbors.disks[color];
  int row;
  int col;
  
  for(row=8; row >=1; row--) {
    ull thisrow = my_neighbor_moves & ROW8;
    for(col=8; thisrow && (col >= 1); col--) {
      if (thisrow & COL8) {
        Move m = { row, col };
        if (FlipDisks(m, &b, color, 0, 0) > 0) {
          legal_moves.push_back(m);
        }
      }
      thisrow >>= 1;
    }
    my_neighbor_moves >>= 8;
  }
  return legal_moves;
}

/*
  return an integer that represents the score of color, which is the difference between 
  the number of disks for color and its opponent
*/
int evaluate_board(Board b, int color) {
  return CountBitsOnBoard(&b, color) - CountBitsOnBoard(&b, OTHERCOLOR(color));
}

/*
  return the best Minimax_Move for color based on the score got after n searching depth. 
  The returned value contains the color disk position (row, col) and final max or min score it gets. 
  It is a recursive function until the terminal state, depth equals 0, reaches.
*/
Minimax_Move minimax_value(Board b, int depth, int color, bool isMax) {
  Minimax_Move move;
  // terminal state
  if (depth == 0) {
    // calculate the score for the color and assign it to the corresponding move
    move.score = evaluate_board(b, color);
    return move;
  }

  // find all legal movment
  std::vector<Move> legal_moves = find_all_legal_moves(b, color);
  if (isMax) {
    // max player will use reducer_max to get the move with maximum score
    cilk::reducer_max<Minimax_Move, max_cmp> best_max_moves;
    cilk_for(int i = 0; i < legal_moves.size(); i++) {
      Board tmpBoard = b;
      int nflips = FlipDisks(legal_moves[i], &tmpBoard, color, 0, 1);
      PlaceOrFlip(legal_moves[i], &tmpBoard, color);
      Minimax_Move next_move = minimax_value(tmpBoard, depth - 1, OTHERCOLOR(color), !isMax);
      next_move.pos = legal_moves[i];
      best_max_moves.calc_max(next_move);
    }
    return best_max_moves.get_value();
  }
  else {
    // min player will use reducer_max to get the move with minimum score
    cilk::reducer_min<Minimax_Move, min_cmp> best_min_moves;
    cilk_for(int i = 0; i < legal_moves.size(); i++) {
      Board tmpBoard = b;
      int nflips = FlipDisks(legal_moves[i], &tmpBoard, color, 0, 1);
      PlaceOrFlip(legal_moves[i], &tmpBoard, color);
      Minimax_Move next_move = minimax_value(tmpBoard, depth - 1, OTHERCOLOR(color), !isMax);
      next_move.pos = legal_moves[i];
      best_min_moves.calc_min(next_move);
    }
    return best_min_moves.get_value();
  }
}

/*
  return an integer:
    1: one of the computer player flips disks
    2: both players have no more legal moves, board has all been filled
  it calles minimax_value function to get the best move on each turn, then move and flip disks
*/
int minimax(Board *b, int depth, int color) {
  Board legal_moves;
  int num_moves = EnumerateLegalMoves(*b, color, &legal_moves);
  // color has no legal moves, check whether opponent has any legal moves
  if (num_moves == 0) {
    int num_moves_op = EnumerateLegalMoves(*b, OTHERCOLOR(color), &legal_moves);
    if (num_moves_op == 0) {
      return 0;
    }
    else {
      Minimax_Move move_op = minimax_value(*b, depth, OTHERCOLOR(color), false);
      printf("Turn: %s\n", (OTHERCOLOR(color) == 0) ? "X_BLACK" : "O_WHITE");
      printf("Move row,column: %d,%d\n", move_op.pos.row, move_op.pos.col);
      int nflips = FlipDisks(move_op.pos, b, OTHERCOLOR(color), 1, 1);
      PlaceOrFlip(move_op.pos, b, OTHERCOLOR(color));
      printf("Computer flipped %d disks\n", nflips);
      PrintBoard(*b);
      return 1;
    }
  }
  Minimax_Move move = minimax_value(*b, depth, color, true);
  printf("Turn: %s\n", (color == 0) ? "X_BLACK" : "O_WHITE");
  printf("Move row,column: %d,%d\n", move.pos.row, move.pos.col);
  int nflips = FlipDisks(move.pos, b,color, 1, 1);
  PlaceOrFlip(move.pos, b, color);
  printf("Computer flipped %d disks\n", nflips);
  PrintBoard(*b);
  return 1;
}

void EndGame(Board b)
{
  int o_score = CountBitsOnBoard(&b,O_WHITE);
  int x_score = CountBitsOnBoard(&b,X_BLACK);
  printf("Game over. \n");
  if (o_score == x_score)  {
    printf("Tie game. Each player has %d disks\n", o_score);
  } else { 
    printf("X has %d disks. O has %d disks. %c wins.\n", x_score, o_score, 
	      (x_score > o_score ? 'X' : 'O'));
  }
}

int main (int argc, const char * argv[]) 
{
  char player1;
  char player2;
  int search_depth1;
  int search_depth2;

  // wait for the user to enter either 'c' or 'h' for two players
  printf("Enter an 'h' or 'c' to specify if player 1 (%c) is a human or computer player: ", diskcolor[X_BLACK + 1]);
  do {
    scanf(" %c", &player1);
  } while (player1 != 'c' && player1 != 'h');
  // enter a search depth for player1 if is a computer player
  if (player1 == 'c') {
    printf("Specifies the search depth between 1 and 60: ");
    scanf(" %d", &search_depth1);
  }
  printf("Enter an 'h' or 'c' to specify if player 2 (%c) is a human or computer player: ", diskcolor[O_WHITE + 1]);
  do {
    scanf(" %c", &player2);
  } while (player2 != 'c' && player2 != 'h');
  // enter a search depth for player2 if is a computer player
  if (player2 == 'c') {
    printf("Specifies the search depth between 1 and 60: ");
    scanf(" %d", &search_depth2);
  }

  Board gameboard = start;
  int move_possible_p1;
  int move_possible_p2;
  PrintBoard(gameboard);
  do {
    if (player1 == 'h') {
      move_possible_p1 = HumanTurn(&gameboard, X_BLACK);
    }
    else {
      move_possible_p1 = minimax(&gameboard, search_depth1, X_BLACK);
    }
    if (player2 == 'h') {
      move_possible_p2 = HumanTurn(&gameboard, O_WHITE);
    }
    else {
      move_possible_p2 = minimax(&gameboard, search_depth2, O_WHITE);
    }
  } while(move_possible_p1 || move_possible_p2);
	
  EndGame(gameboard);
	
  return 0;
}
