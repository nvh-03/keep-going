import pygame
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# SETTINGS
# ==========================================

WIDTH = 600
HEIGHT = 600
BLOCK_SIZE = 30

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

LEARNING_RATE = 0.001
GAMMA = 0.9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# NEURAL NETWORK
# ==========================================

class LinearQNet(nn.Module):

    def __init__(
        self,
        input_size=11,
        hidden_size=256,
        output_size=3
    ):

        super().__init__()

        self.linear1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.linear2 = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(self, x):

        x = torch.relu(
            self.linear1(x)
        )

        x = self.linear2(x)

        return x


    def save(self, file_name="snake_model.pth"):

        torch.save(
            self.state_dict(),
            file_name
        )


# ==========================================
# TRAINER
# ==========================================

class QTrainer:

    def __init__(
        self,
        model,
        lr,
        gamma
    ):

        self.model = model

        self.lr = lr

        self.gamma = gamma

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr
        )

        self.criterion = nn.MSELoss()


    def train_step(

        self,

        state,

        action,

        reward,

        next_state,

        done

    ):

        # Convert to Tensor

        state = torch.tensor(
            np.array(state),
            dtype=torch.float
        )

        next_state = torch.tensor(
            np.array(next_state),
            dtype=torch.float
        )

        action = torch.tensor(
            np.array(action),
            dtype=torch.long
        )

        reward = torch.tensor(
            np.array(reward),
            dtype=torch.float
        )


        # Nếu chỉ có 1 state

        if len(state.shape) == 1:

            state = torch.unsqueeze(
                state,
                0
            )

            next_state = torch.unsqueeze(
                next_state,
                0
            )

            action = torch.unsqueeze(
                action,
                0
            )

            reward = torch.unsqueeze(
                reward,
                0
            )

            done = (done,)


        # Prediction

        pred = self.model(
            state
        )


        target = pred.clone()


        for index in range(len(done)):


            Q_new = reward[index]


            if not done[index]:

                Q_new = reward[index] + (

                    self.gamma

                    *

                    torch.max(
                        self.model(
                            next_state[index]
                        )
                    )

                )


            action_index = torch.argmax(
                action[index]
            ).item()


            target[index][
                action_index
            ] = Q_new


        # Train

        self.optimizer.zero_grad()


        loss = self.criterion(
            pred,
            target
        )


        loss.backward()


        self.optimizer.step()


# ==========================================
# SNAKE GAME
# ==========================================

class SnakeGame:


    def __init__(self):

        pygame.init()


        self.screen = pygame.display.set_mode(

            (
                WIDTH,
                HEIGHT
            )

        )


        pygame.display.set_caption(
            "Snake AI - Deep Q Learning"
        )


        self.clock = pygame.time.Clock()


        self.font = pygame.font.SysFont(
            "arial",
            25
        )


        self.reset()


    # ======================================
    # RESET GAME
    # ======================================

    def reset(self):


        self.direction = "right"


        self.snake = [

            [7, 10],

            [8, 10],

            [9, 10]

        ]


        self.head = self.snake[-1]


        self.score = 0


        self.frame_iteration = 0


        self.create_apple()


    # ======================================
    # CREATE APPLE
    # ======================================

    def create_apple(self):


        while True:


            apple = [

                random.randint(
                    0,
                    19
                ),

                random.randint(
                    0,
                    19
                )

            ]


            if apple not in self.snake:


                self.apple = apple


                break


    # ======================================
    # GAME STEP
    # ======================================

    def play_step(
        self,
        action
    ):


        self.frame_iteration += 1


        # Quit event

        for event in pygame.event.get():


            if event.type == pygame.QUIT:


                pygame.quit()


                quit()


        # Move snake

        self.move(
            action
        )


        reward = 0


        game_over = False


        # Check collision

        if self.is_collision():


            reward = -10


            game_over = True


            return (
                reward,
                game_over,
                self.score
            )


        # Limit steps
        # Prevent snake from moving forever

        if self.frame_iteration > 100 * len(self.snake):


            reward = -10


            game_over = True


            return (
                reward,
                game_over,
                self.score
            )


        # Eat apple

        if self.head == self.apple:


            self.score += 1


            reward = 10


            self.create_apple()


        else:


            # Remove tail

            self.snake.pop(0)


        # Draw game

        self.update_ui()


        # Game speed

        self.clock.tick(
            50
        )


        return (

            reward,

            game_over,

            self.score

        )


    # ======================================
    # COLLISION
    # ======================================

    def is_collision(
        self,
        point=None
    ):


        if point is None:


            point = self.head


        # Wall collision

        if point[0] < 0:


            return True


        if point[0] > 19:


            return True


        if point[1] < 0:


            return True


        if point[1] > 19:


            return True


        # Body collision

        if point in self.snake[:-1]:


            return True


        return False


    # ======================================
    # MOVE
    # ======================================

    def move(
        self,
        action
    ):


        directions = [

            "right",

            "down",

            "left",

            "up"

        ]


        index = directions.index(
            self.direction
        )


        # =================================
        # ACTION
        #
        # [1,0,0] = Straight
        #
        # [0,1,0] = Right
        #
        # [0,0,1] = Left
        # =================================


        if action[0] == 1:


            new_direction = directions[
                index
            ]


        elif action[1] == 1:


            index = (
                index + 1
            ) % 4


            new_direction = directions[
                index
            ]


        else:


            index = (
                index - 1
            ) % 4


            new_direction = directions[
                index
            ]


        self.direction = new_direction


        x = self.head[0]


        y = self.head[1]


        # Move

        if self.direction == "right":


            x += 1


        elif self.direction == "left":


            x -= 1


        elif self.direction == "up":


            y -= 1


        elif self.direction == "down":


            y += 1


        # New head

        self.head = [

            x,

            y

        ]


        self.snake.append(
            self.head
        )


    # ======================================
    # DRAW UI
    # ======================================

    def update_ui(self):


        BLACK = (
            0,
            0,
            0
        )


        GREEN = (
            0,
            200,
            0
        )


        RED = (
            255,
            0,
            0
        )


        WHITE = (
            255,
            255,
            255
        )


        # Background

        self.screen.fill(
            BLACK
        )


        # Draw snake

        for block in self.snake:


            pygame.draw.rect(

                self.screen,

                GREEN,

                (

                    block[0]
                    *
                    BLOCK_SIZE,

                    block[1]
                    *
                    BLOCK_SIZE,

                    BLOCK_SIZE,

                    BLOCK_SIZE

                )

            )


        # Draw apple

        pygame.draw.rect(

            self.screen,

            RED,

            (

                self.apple[0]
                *
                BLOCK_SIZE,

                self.apple[1]
                *
                BLOCK_SIZE,

                BLOCK_SIZE,

                BLOCK_SIZE

            )

        )


        # Score

        score_text = self.font.render(

            "Score: "
            +
            str(self.score),

            True,

            WHITE

        )


        self.screen.blit(

            score_text,

            (

                10,

                10

            )

        )


        pygame.display.flip()


# ==========================================
# AI AGENT
# ==========================================

class Agent:


    def __init__(self):


        self.n_games = 0


        self.epsilon = 0


        self.gamma = GAMMA


        self.memory = deque(

            maxlen=MAX_MEMORY

        )


        # Neural Network

        self.model = LinearQNet(

            input_size=11,

            hidden_size=256,

            output_size=3

        )


        # Trainer

        self.trainer = QTrainer(

            self.model,

            lr=LEARNING_RATE,

            gamma=self.gamma

        )


    # ======================================
    # GET STATE
    # ======================================

    def get_state(
        self,
        game
    ):


        head = game.head


        # Points around head

        point_left = [

            head[0] - 1,

            head[1]

        ]


        point_right = [

            head[0] + 1,

            head[1]

        ]


        point_up = [

            head[0],

            head[1] - 1

        ]


        point_down = [

            head[0],

            head[1] + 1

        ]


        # Current direction

        dir_left = (

            game.direction == "left"

        )


        dir_right = (

            game.direction == "right"

        )


        dir_up = (

            game.direction == "up"

        )


        dir_down = (

            game.direction == "down"

        )


        # =================================
        # STATE
        # =================================


        state = [


            # -------------------------
            # Danger straight
            # -------------------------

            (

                dir_right

                and

                game.is_collision(
                    point_right
                )

            )

            or

            (

                dir_left

                and

                game.is_collision(
                    point_left
                )

            )

            or

            (

                dir_up

                and

                game.is_collision(
                    point_up
                )

            )

            or

            (

                dir_down

                and

                game.is_collision(
                    point_down
                )

            ),


            # -------------------------
            # Danger right
            # -------------------------

            (

                dir_up

                and

                game.is_collision(
                    point_right
                )

            )

            or

            (

                dir_down

                and

                game.is_collision(
                    point_left
                )

            )

            or

            (

                dir_left

                and

                game.is_collision(
                    point_up
                )

            )

            or

            (

                dir_right

                and

                game.is_collision(
                    point_down
                )

            ),


            # -------------------------
            # Danger left
            # -------------------------

            (

                dir_down

                and

                game.is_collision(
                    point_right
                )

            )

            or

            (

                dir_up

                and

                game.is_collision(
                    point_left
                )

            )

            or

            (

                dir_right

                and

                game.is_collision(
                    point_up
                )

            )

            or

            (

                dir_left

                and

                game.is_collision(
                    point_down
                )

            ),


            # -------------------------
            # Direction
            # -------------------------

            dir_left,

            dir_right,

            dir_up,

            dir_down,


            # -------------------------
            # Apple
            # -------------------------

            game.apple[0]
            <
            head[0],


            game.apple[0]
            >
            head[0],


            game.apple[1]
            <
            head[1],


            game.apple[1]
            >
            head[1]

        ]


        return np.array(

            state,

            dtype=int

        )


    # ======================================
    # SAVE MEMORY
    # ======================================

    def remember(

        self,

        state,

        action,

        reward,

        next_state,

        done

    ):


        self.memory.append(

            (

                state,

                action,

                reward,

                next_state,

                done

            )

        )


    # ======================================
    # TRAIN LONG MEMORY
    # ======================================

    def train_long_memory(self):


        if len(
            self.memory
        ) > BATCH_SIZE:


            mini_sample = random.sample(

                self.memory,

                BATCH_SIZE

            )


        else:


            mini_sample = self.memory


        states = []


        actions = []


        rewards = []


        next_states = []


        dones = []


        for item in mini_sample:


            states.append(
                item[0]
            )


            actions.append(
                item[1]
            )


            rewards.append(
                item[2]
            )


            next_states.append(
                item[3]
            )


            dones.append(
                item[4]
            )


        self.trainer.train_step(

            states,

            actions,

            rewards,

            next_states,

            dones

        )


    # ======================================
    # TRAIN SHORT MEMORY
    # ======================================

    def train_short_memory(

        self,

        state,

        action,

        reward,

        next_state,

        done

    ):


        self.trainer.train_step(

            state,

            action,

            reward,

            next_state,

            done

        )


    # ======================================
    # GET ACTION
    # ======================================

    def get_action(
        self,
        state
    ):


        # Exploration rate

        self.epsilon = max(

            0,

            80 - self.n_games

        )


        final_move = [

            0,

            0,

            0

        ]


        # ===============================
        # RANDOM ACTION
        # ===============================

        if random.randint(
            0,
            200
        ) < self.epsilon:


            move = random.randint(

                0,

                2

            )


            final_move[
                move
            ] = 1


        # ===============================
        # AI ACTION
        # ===============================

        else:


            state_tensor = torch.tensor(

                state,

                dtype=torch.float

            )


            with torch.no_grad():


                prediction = self.model(

                    state_tensor

                )


            move = torch.argmax(

                prediction

            ).item()


            final_move[
                move
            ] = 1


        return final_move


# ==========================================
# TRAIN FUNCTION
# ==========================================

def train():


    agent = Agent()


    game = SnakeGame()


    record = 0


    print(
        "Start training Snake AI..."
    )


    while True:


        # =================================
        # 1. GET STATE
        # =================================

        state_old = agent.get_state(
            game
        )


        # =================================
        # 2. GET ACTION
        # =================================

        final_move = agent.get_action(
            state_old
        )


        # =================================
        # 3. MOVE SNAKE
        # =================================

        reward, done, score = game.play_step(

            final_move

        )


        # =================================
        # 4. GET NEW STATE
        # =================================

        state_new = agent.get_state(
            game
        )


        # =================================
        # 5. TRAIN SHORT MEMORY
        # =================================

        agent.train_short_memory(

            state_old,

            final_move,

            reward,

            state_new,

            done

        )


        # =================================
        # 6. REMEMBER
        # =================================

        agent.remember(

            state_old,

            final_move,

            reward,

            state_new,

            done

        )


        # =================================
        # 7. GAME OVER
        # =================================

        if done:


            game.reset()


            agent.n_games += 1


            agent.train_long_memory()


            # New record

            if score > record:


                record = score


                agent.model.save(
                    "snake_model.pth"
                )


            # Print result

            print(

                "Game:",

                agent.n_games,

                "| Score:",

                score,

                "| Record:",

                record,

                "| Memory:",

                len(agent.memory)

            )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train()