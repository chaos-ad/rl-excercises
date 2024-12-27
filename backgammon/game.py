import sys
import copy 
import random
import itertools
import math
import unittest
import pandas as pd
import numpy as np
import logging
import torch
import tqdm.auto as tqdm
from pathlib import Path

from typing import *
from enum import Enum
from dataclasses import dataclass

##############################################################################

logger = logging.getLogger("backgammon")

##############################################################################

class Game:
    class Step(Enum):
        IDLE = "idle"
        ROLL = "roll"
        TURN = "trun"
        FINISHED = "finished"

    def __init__(self, seed: int | None = None, verbose = False):
        self.randgen = np.random.Generator(bit_generator=np.random.MT19937(seed=seed))
        self.board = np.zeros(24, dtype=int)
        self.board[0] = 15
        self.board[12] = -15
        self.dice = [0, 0]
        self.home = [0, 0]
        self.pturn = 0
        self.t = 0
        self.head_moves = 0
        self.valid_moves = None
        self.step : Game.Step = Game.Step.IDLE
        self.loglevel = logging.INFO if verbose else logging.DEBUG
        logger.log(self.loglevel, f"new game started, seed={self.seed}")

    @property
    def seed(self) -> int:
        return self.randgen.bit_generator.seed_seq.entropy

    @property
    def _opponent(self) -> int:
        return 1 if self.pturn == 0 else 0
    
    @property
    def _cur_player(self) -> int:
        return self.pturn
    
    def _randint(self, lo: int = 0, hi: int = 6) -> int:
        return int(self.randgen.integers(lo,hi,endpoint=True))
    
    def _randroll(self) -> int:
        return self._randint(1,6)

    def _randfloat(self) -> float:
        return self.randgen.random()
    
    def _randchoice(self, values: List[Any]) -> Any:
        return values[self._randint(hi=len(values)-1)] if len(values) > 0 else None

    def _get_next_pos(self, pos: int, steps: int) -> int:
        dst_pos = pos + steps
        if dst_pos > 24:
            dst_pos = dst_pos % 25 + 1
        if dst_pos < 1:
            dst_pos = 24 + dst_pos
        return dst_pos
    
    def _get_head(self, player: Optional[int] = None) -> int:
        player = player if player is not None else self.pturn
        return 1 if player == 0 else 13
    
    def _get_user_sign(self, player: Optional[int] = None) -> int:
        player = player if player is not None else self.pturn
        return 1 if player == 0 else -1

    def _can_move_home(self):
        counter = self.home[self.pturn]
        home = range(19, 25) if self.pturn == 0 else range(7, 13)
        for pos in home:
            if self._has_checkers(pos, player=self._cur_player):
                counter += self._get_checkers(pos, player=self._cur_player)
        return counter == 15

    def _is_move_home(self, pos: int, steps: int) -> bool:
        dst_pos = self._get_next_pos(pos, steps)
        return (self.pturn == 0 and dst_pos < pos) or \
               (self.pturn == 1 and pos <= 12 and dst_pos > 12)

    def _has_checkers(self, pos: int, player: Optional[int] = None):
        return self._get_checkers(pos, player) > 0
    
    def _get_checkers(self, pos: int, player: Optional[int] = None):
        player = player if player is not None else self.pturn
        return self._get_user_sign(player) * self.board[pos-1]
    
    def _find_prime(self, pos: int) -> Optional[Tuple[int, int]]:
        seq = 1
        result = [pos, pos]
        next_ptr = self._get_next_pos(pos, 1)
        while seq < 6 and self._has_checkers(next_ptr, player=self._cur_player):
            seq += 1
            result[1] = next_ptr
            next_ptr = self._get_next_pos(next_ptr, 1)
        prev_ptr = self._get_next_pos(pos, -1)
        while seq < 6 and self._has_checkers(prev_ptr, player=self._cur_player):
            seq += 1
            result[0] = prev_ptr
            prev_ptr = self._get_next_pos(prev_ptr, -1)
        return tuple(result) if seq == 6 else None
    
    def _is_blocking_prime(self, dst_pos: int) -> bool:
        prime_range = self._find_prime(dst_pos)
        if prime_range:
            prime_end_pos = prime_range[1]
            for step in range(1, 24):
                search_pos = self._get_next_pos(prime_end_pos, step)
                if self._get_head(self._opponent) == search_pos:
                    return True # we reached other player's home without finding it's checkers
                if self._has_checkers(pos=search_pos, player=self._opponent):
                    break
        return False
    
    def _check_move(self, pos: int, steps: int):
        dst_pos = self._get_next_pos(pos, steps)
        if self.step != Game.Step.TURN:
            raise RuntimeError("invalid action")
        if not (1 <= pos <= 24):
            raise RuntimeError("invalid position")
        if not self._has_checkers(pos, player=self._cur_player):
            raise RuntimeError(f"no checkers at position {pos}")
        if self._has_checkers(dst_pos, player=self._opponent):
            raise RuntimeError(f"can't move to position {dst_pos}")
        if steps not in self.dice:
            raise RuntimeError(f"no dice with value {steps}")
        if self._is_move_home(pos, steps):
            if not self._can_move_home():
                raise RuntimeError(f"not all checkers are at finishing table")
        if self._get_head() == pos and self.head_moves > 0:
            if not (self.head_moves == 1 and self.dice[0] == self.dice[1] and self.t < 2):
                raise RuntimeError(f"can't make any more head moves")
        if self._is_blocking_prime(dst_pos):
            raise RuntimeError(f"can't form a blocking prime")
        # TODO: If player can play one number but not both, they must play the higher one

    def _render_player(self, player: Optional[int] = None, lower: bool = True) -> str:
        player = player if player is not None else self.pturn
        result = "O" if player == 0 else "X"
        return result.lower() if lower else result

    def _is_valid_move(self, pos: int, steps: int) -> bool:
        try:
            self._check_move(pos, steps)
            return True
        except RuntimeError as e:
            return False
    
    def _enum_valid_moves(self) -> Iterator[Tuple[int, int]]:
        for pos in range(1, 25):
            if self._has_checkers(pos, player=self._cur_player):
                for steps in range(1, 7):
                    if self._is_valid_move(pos, steps):
                        yield (pos, steps)

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        if self.valid_moves is None:
            self.valid_moves = list(self._enum_valid_moves())
        return self.valid_moves
    
    def has_valid_moves(self):
        return len(self.get_valid_moves()) > 0

    def start(self, d1: int = 0, d2: int = 0) -> "Game":
        if self.step != Game.Step.IDLE:
            raise RuntimeError("invalid action")

        self.dice = [d1 or self._randroll(), d2 or self._randroll()]
        while self.dice[0] == self.dice[1]:
            self.dice = [self._randroll(), self._randroll()]

        self.step = Game.Step.ROLL
        if self.dice[0] > self.dice[1]:
            self.pturn = 0
        else: # self.dice[0] < self.dice[1]:
            self.pturn = 1
        return self

    def roll(self, d1: int = 0, d2: int = 0) -> "Game":
        if self.step != Game.Step.ROLL:
            raise RuntimeError("invalid action")
        self.dice = [d1 or self._randroll(), d2 or self._randroll()]
        logger.log(self.loglevel, f"t={self.t}, p={self._render_player()} rolls {self.dice}")
        if self.dice[0] == self.dice[1]:
            self.dice += self.dice
        self.step = Game.Step.TURN
        return self

    def turn(self, pos: int, steps: int) -> "Game":
        dst_pos = self._get_next_pos(pos, steps)
        self._check_move(pos, steps)
        if self._is_move_home(pos, steps):
            logger.log(self.loglevel, f"t={self.t}, p={self._render_player()} moves: {pos}->HOME")
            self.board[pos-1] -= self._get_user_sign()
            self.home[self.pturn] += 1
        else:
            logger.log(self.loglevel, f"t={self.t}, p={self._render_player()} moves: {pos}-({steps})->{dst_pos}")
            self.board[pos-1] -= self._get_user_sign()
            self.board[dst_pos-1] += self._get_user_sign()

        if self._get_head() == pos:
            self.head_moves += 1

        if len(self.dice) > 2:
            self.dice.pop(self.dice.index(steps, -1))
        else:
            self.dice[self.dice.index(steps)] = 0

        if self.home[self.pturn] == 15:
            logger.log(self.loglevel, f"t={self.t}, game finished, p={self._render_player()} wins")
            self.step = Game.Step.FINISHED
        elif self.dice[0] == 0 and self.dice[1] == 0:
            self.step = Game.Step.ROLL
            self.pturn = self._opponent
            self.t += 1
            self.head_moves = 0

        self.valid_moves = None
        return self
    
    def is_finished(self):
        return self.step == Game.Step.FINISHED
    
    def skip(self) -> "Game":
        if self.step != Game.Step.TURN:
            raise RuntimeError("invalid action")
        
        if self.has_valid_moves():
            raise RuntimeError("skip only possible when there's no moves")
        else:
            logger.log(self.loglevel, f"t={self.t}, p={self._render_player()} has no eligible moves, skipping")
            self.dice = [0, 0]
            self.step = Game.Step.ROLL
            self.pturn = (self.pturn + 1) % 2
            self.t += 1
            self.head_moves = 0
            
        self.valid_moves = None
        return self

    def __repr__(self):
        template = """
        |{oha}| 24 | 23 | 22 | 21 | 20 | 19 |{xst}| 18 | 17 | 16 | 15 | 14 | 13 |{xho}|
        |{ohb}|-----------------------------|     |-----------------------------|{xhn}|
        |{ohc}|{x1}|{w1}|{v1}|{u1}|{t1}|{s1}|     |{r1}|{q1}|{p1}|{o1}|{n1}|{m1}|{xhm}|
        |{ohd}|{x2}|{w2}|{v2}|{u2}|{t2}|{s2}|     |{r2}|{q2}|{p2}|{o2}|{n2}|{m2}|{xhl}|
        |{ohe}|{x3}|{w3}|{v3}|{u3}|{t3}|{s3}|     |{r3}|{q3}|{p3}|{o3}|{n3}|{m3}|{xhk}|
        |{ohf}|{x4}|{w4}|{v4}|{u4}|{t4}|{s4}|     |{r4}|{q4}|{p4}|{o4}|{n4}|{m4}|{xhj}|
        |{ohg}|{x5}|{w5}|{v5}|{u5}|{t5}|{s5}|     |{r5}|{q5}|{p5}|{o5}|{n5}|{m5}|{xhi}|
        |{ohh}|-----------------------------|{dcs}|-----------------------------|{xhh}|
        |{ohi}|{a5}|{b5}|{c5}|{d5}|{e5}|{f5}|     |{g5}|{h5}|{i5}|{j5}|{k5}|{l5}|{xhg}|
        |{ohj}|{a4}|{b4}|{c4}|{d4}|{e4}|{f4}|     |{g4}|{h4}|{i4}|{j4}|{k4}|{l4}|{xhf}|
        |{ohk}|{a3}|{b3}|{c3}|{d3}|{e3}|{f3}|     |{g3}|{h3}|{i3}|{j3}|{k3}|{l3}|{xhe}|
        |{ohl}|{a2}|{b2}|{c2}|{d2}|{e2}|{f2}|     |{g2}|{h2}|{i2}|{j2}|{k2}|{l2}|{xhd}|
        |{ohm}|{a1}|{b1}|{c1}|{d1}|{e1}|{f1}|     |{g1}|{h1}|{i1}|{j1}|{k1}|{l1}|{xhc}|
        |{ohn}|-----------------------------|     |-----------------------------|{xhb}|
        |{oho}| 01 | 02 | 03 | 04 | 05 | 06 |{ost}| 07 | 08 | 09 | 10 | 11 | 12 |{xha}|
        """

        pixels = {}
        for pos in range(1,25):
            x = chr(ord("a") + pos - 1)
            checkers = abs(int(self.board[pos-1]))
            pixel = self._render_player(player = 0 if self.board[pos-1] > 0 else 1)
            for y in range(1, 6):
                if checkers > 0:
                    if y < 5 or checkers == 1:
                        pixels[f"{x}{y}"] = (f"  {pixel} ")
                    else: # y == 5 and checkers > 1:
                        pixels[f"{x}{y}"] = f"({checkers}".rjust(3) + ")"
                    checkers -= 1
                else:
                    pixels[f"{x}{y}"] = "    "
        pixels["dcs"] = f" {self.dice[0] or ' '}:{self.dice[1] or ' '} "

        for player in [0, 1]:
            pixel = self._render_player(player=player)
            for idx in range(15):
                h = pixel + "h" + chr(ord("a") + idx)
                pixels[h] = f"  {pixel}  " if self.home[player] > idx else "     "

        pixels["ost"] = "     "
        pixels["xst"] = "     "
        if self.step == Game.Step.ROLL:
            pixels["ost" if self.pturn == 0 else "xst"] = " ROL "
        elif self.step == Game.Step.TURN:
            pixels["ost" if self.pturn == 0 else "xst"] = f" ({len([d for d in self.dice if d])}) "
        elif self.step == Game.Step.FINISHED:
            pixels["ost" if self.pturn == 0 else "xst"] = " WIN "

        return template.format(**pixels)

##############################################################################

class GameTests(unittest.TestCase):

    def test_heads(self):
        g = Game(seed=42).start(6,1)
        g.roll(3,3).turn(1,3).turn(1,3) # < can make 2 moves from head on doubles
        with self.assertRaises(RuntimeError):
            g.turn(1,3)
        g.turn(4,3).turn(4,3)
        g.roll(1,2).turn(13,1).turn(14,2) # p2
        g.roll(2,2).turn(1,2)
        with self.assertRaises(RuntimeError):
            g.turn(1, 2)
        g.turn(3,2)
        with self.assertRaises(RuntimeError):
            g.turn(1, 2)
        g.turn(5,2)

    def test_skips(self):
        g = Game(seed=42).start(6,1)
        g.roll(6,1).turn(1, 6).turn(7, 1)
        g.roll(3,2).turn(13, 3).turn(16, 2)
        g.roll(2,2).turn(1, 2).turn(3, 2).turn(5, 2).turn(8, 2)
        g.roll(6,1).turn(13,6).turn(18,1)
        g.roll(6,6).turn(10, 6).turn(16,6).turn(1, 6) # < there's no moves for red here
        with self.assertRaises(RuntimeError):
            g.turn(1, 6)
        g.skip()

    def test_blocking_primes(self):
        g = Game(seed=42).start(6,1)
        g.roll(1,1).turn(1,1).turn(1,1).turn(2,1).turn(3,1)
        g.roll(6,5).turn(13,6).turn(19,5)
        g.roll(2,3).turn(1,2).turn(3,3)
        g.roll(4,2).turn(13,4).turn(17,2)
        g.roll(3,1).turn(1,1).turn(2,3)
        g.roll(3,6).turn(13,3).turn(16,6)
        g.roll(1,2)
        with self.assertRaises(RuntimeError):
            g.turn(1,2)

    def test_nonblocking_primes(self):
        g = Game(seed=42).start(6,1)
        g.roll(1,1).turn(1,1).turn(1,1).turn(2,1).turn(3,1)
        g.roll(6,5).turn(13,6).turn(19,5)
        g.roll(2,3).turn(1,2).turn(3,3)
        g.roll(4,2).turn(13,4).turn(17,2)
        g.roll(3,1).turn(1,1).turn(2,3)
        g.roll(3,6).turn(24,3).turn(3,6)
        g.roll(1,2).turn(1,2) # < valid move, since x is ahead

##############################################################################

class BasePlayer:
    def play_turn(self, game: Game) -> bool:
        pass

class RandomPlayer(BasePlayer):
    def play_turn(self, game: Game) -> bool:
        actions = game.get_valid_moves()
        if len(actions) > 0:
            pos, steps = game._randchoice(actions)
            game.turn(pos, steps)
            return True
        else:
            game.skip()
            return False

class LazyPlayer(BasePlayer):
    def play_turn(self, game: Game) -> bool:
        actions = game.get_valid_moves()
        if actions:
            pos, steps = actions[0]
            game.turn(pos, steps)
            return True
        else:
            game.skip()
            return False

##############################################################################

class AutoGame(Game):
    def __init__(self, player1: BasePlayer | None = None, player2: BasePlayer | None = None, start: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_player1 = player1
        self.auto_player2 = player2
        if start:
            self.start()

    def _automate(self) -> "AutoGame":
        while not self.is_finished():
            if self.step == Game.Step.ROLL:
                self.roll()
            if self.step == Game.Step.TURN:
                if not self.has_valid_moves():
                    self.skip()
                else:
                    player = self.auto_player1 if (self.pturn == 0) else self.auto_player2
                    if player:
                        player.play_turn(self)
                    else:
                        break
        return self
   
    def start(self, *args, **kwargs) -> "AutoGame":
        super().start(*args, **kwargs)
        return self._automate()

    def turn(self, *args, **kwargs) -> "AutoGame":
        super().turn(*args, **kwargs)
        return self._automate()

    def play_sequence(self, turns: List[Tuple[int, int]]) -> "AutoGame":
        for (pos, steps) in turns:
            self.turn(pos, steps)
        return self

##############################################################################

class AutoGameTests(unittest.TestCase):

    def test_auto(self):
        AutoGame(seed=14).play_sequence([(1,4), (5,2), (13,6), (19,3), (1,6), (7,2), (13,3), (22,1), (1,5)])
        with self.assertRaises(RuntimeError):
            AutoGame(seed=14).play_sequence([(1,4), (5,2), (13,6), (19,3), (1,6), (7,2), (13,3), (22,1), (1,5), (1,5)])

    def test_autoplayer(self):
        AutoGame(seed=14, player2=LazyPlayer()).play_sequence([(1,4), (5,2), (1,6), (7,2), (1,5), (6,5), (7,5)])
        AutoGame(seed=14, player1=LazyPlayer()).play_sequence([(13,6), (19,3), (13,3), (22,1), (13,2), (23,5)])
        self.assertEqual(AutoGame(seed=324, player1=LazyPlayer(), player2=LazyPlayer()).is_finished(), True)
        self.assertEqual(AutoGame(seed=324, start=False, player1=LazyPlayer(), player2=LazyPlayer()).is_finished(), False)

##############################################################################

@dataclass
class Result:
    winner: int
    turns: int
    reward: int

def summarize(game: Game) -> Result:
    assert game.is_finished()
    reward =  (2 if game.home[game._opponent] == 0 else 1)
    reward *= (1 if game.pturn == 0 else -1)
    result = Result(
        winner = int(game.pturn == 0),
        turns = game.t,
        reward = reward
    )
    return result

def simulate(game: Game, player1: BasePlayer, player2: BasePlayer | None = None) -> Result:
    while not game.is_finished():
        player = player1 if (game.pturn == 0) else player2
        if player:
            player.play_turn(game)
    return summarize(game)

##############################################################################
