from pyswagger import App
from pyswagger.contrib.client.requests import Client

import random
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optiom
import numpy as np

from ReplayMemory import ReplayMemory
from DQN import DQN


class QLearning:
    def __init__(self, reward_decay=0.9, initial_epsilon=0.5):
        self.action_dim = 25
        self.action_step = np.log(3/0.3) / self.action_dim
        self.gamma = reward_decay
        self.eps_start = initial_epsilon
        self.eps_end = 0.05
        self.eps_decay = 100
        self.step = 0
        self.dqn_1 = DQN(1, 10, self.action_dim)
        self.dqn_2 = DQN(1, 10, self.action_dim)
        self.memory = ReplayMemory(48)
        self.batch_size = 16


    def train_dqn(self, model):
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        if batch_size == 0:
            return
        samples = self.memory.sample(batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        loss = 0
        criterion = nn.MSELoss()
        optimizer = optiom.Adam(model.parameters())
        for i in range(batch_size):
            states.append(samples[i].state)
            actions.append(samples[i].action)
            next_states.append(samples[i].next_state)
            rewards.append(samples[i].reward)

        states = torch.tensor(states).reshape(-1, 10, 1)
        next_states = torch.tensor(next_states).reshape(-1, 10, 1)

        model.eval()
        next_output = model(next_states)
        model.train()
        output = model(states)
        for i in range(batch_size):
            t = rewards[i] + torch.max(next_output[i]) * self.gamma
            loss += criterion(output[i][actions[i]], t)

        loss /= batch_size
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def choose_action(self, state, model):
        # action selection
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.step / self.eps_decay)
        self.step += 1
        if np.random.uniform() > eps:
            # choose best action
            model.eval()
            values = model(state).detach().numpy().reshape(-1)
            action = np.argmax(values)
        else:
            # choose random action
            action = random.randint(0, self.action_dim - 1)
        return action

    def action2number(self, action, last_gn):
        return np.exp(self.action_step * action) * 0.3 * last_gn


lastState = None
lastAction = None
lastNumber = None

def getState(goldenNumberList):
    gn = np.array(goldenNumberList[-10:])
    gn = gn / gn[-1]
    if len(gn) < 10:
        gn2 = np.zeros(10)
        gn2[-len(gn):] = gn
        gn = gn2
    gn = torch.tensor(gn).float().reshape(1, 10, 1)
    return gn

def getRewards(last_gn, numberList, last_num):
    reward = 0
    numbers = []
    num_bot = len(numberList)

    for i in range(num_bot):
        numbers.append(numberList[i].number1)
        numbers.append(numberList[i].number2)
    numbers = np.array(numbers)
    numbers = np.abs(numbers - last_gn)
    numbers.sort()
    num = np.abs(last_num - last_gn)
    rank = np.sum(numbers < num)
    if rank == 0:
        reward = num_bot
    elif rank == num_bot - 1:
        reward = -2
    else:
        reward = num_bot * 2**(-float(rank))
    return reward

RL = QLearning()

def GeneratePredictionNumbers(goldenNumberList, numberList):
    global lastAction
    global lastState
    global lastNumber

    if len(goldenNumberList) < 1:
        return 19.0, 19.0
    state = getState(goldenNumberList)

    RL.train_dqn(RL.dqn_1)
    RL.train_dqn(RL.dqn_2)

    action_1 = RL.choose_action(state, RL.dqn_1)
    action_2 = RL.choose_action(state, RL.dqn_2)

    number1 = RL.action2number(action_1, goldenNumberList[-1])
    number2 = RL.action2number(action_2, goldenNumberList[-1])

    lastAction = [action_1, action_2]
    lastNumber = [number1, number2]
    lastState = state

    if lastState is not None:
        for i in range(2):
            reward = getRewards(goldenNumberList[-1], numberList, lastNumber[i])
            RL.memory.push(lastState.tolist(), lastAction[i], state.tolist(), reward)

    return number1, number2

# Init swagger client
host = 'https://goldennumber.aiedu.msra.cn/'
jsonpath = '/swagger/v1/swagger.json'
app = App._create_(host + jsonpath)
client = Client()

def main(roomId):
    if roomId is None:
        # Input the roomid if there is no roomid in args
        roomId = input("Input room id: ")
        try:
            roomId = int(roomId)
        except:
            roomId = 0
            print('Parse room id failed, default join in to room 0')

    userInfoFile = "userinfo0.txt"
    userId = None
    nickName = None
    try:
        # Use an exist player
        with open(userInfoFile) as f:
            userId, nickName = f.read().split(',')[:2]
        print('Use an exist player: ' + nickName + '  Id: ' + userId)
    except:
        # Create a new player
        userResp = client.request(
            app.op['NewUser'](
                nickName='AI Player ' + str(random.randint(0, 9999))
            ))
        assert userResp.status == 200
        user = userResp.data
        userId = user.userId
        nickName = user.nickName
        print('Create a new player: ' + nickName + '  Id: ' + userId)

        with open(userInfoFile, "w") as f:
            f.write("%s,%s" % (userId, nickName))

    print('Room id: ' + str(roomId))

    while True:
        stateResp = client.request(
            app.op['State'](
                uid=userId,
                roomid=roomId
            ))
        if stateResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        state = stateResp.data

        if state.state == 2:
            print('The game has finished')
            break

        if state.state == 1:
            print('The game has not started, query again after 1 second')
            time.sleep(1)
            continue

        if state.hasSubmitted:
            print('Already submitted this round, wait for next round')
            if state.maxUserCount == 0:
                time.sleep(state.leftTime + 1)
            else:
                # One round can be finished when all players submitted their numbers if the room have set the max count of users, need to check the state every second.
                time.sleep(1)
            continue

        print('\r\nThis is round ' + str(state.finishedRoundCount + 1))

        todayGoldenListResp = client.request(
            app.op['TodayGoldenList'](
                roomid=roomId
            ))
        if todayGoldenListResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        todayGoldenList = todayGoldenListResp.data
        if len(todayGoldenList.goldenNumberList) != 0:
            print('Last golden number is: ' + str(todayGoldenList.goldenNumberList[-1]))

        todayNumbersResp = client.request(
            app.op['TodayNumbers'](
                roomid=roomId,
                roundCount=1
            ))
        if todayNumbersResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        todayNumbers = todayNumbersResp.data

        lastRoundResp = client.request(
            app.op['History'](
                roomid=roomId,
                count=1
            ))
        if lastRoundResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        lastScore = 0
        if len(lastRoundResp.data.rounds) > 0:
            scoreArray = [user for user in lastRoundResp.data.rounds[0].userNumbers if user.userId == userId]
            if len(scoreArray) == 1:
                lastScore = scoreArray[0].score
        print('Last round score: {}'.format(lastScore))
        start = time.time()
        number1, number2 = GeneratePredictionNumbers(todayGoldenList.goldenNumberList, todayNumbers.numberList)
        print("time cost:", time.time() - start)

        if (state.numbers == 2):
            submitRsp = client.request(
                app.op['Submit'](
                    uid=userId,
                    rid=state.roundId,
                    n1=str(number1),
                    n2=str(number2)
                ))
            if submitRsp.status == 200:
                print('You submit numbers: ' + str(number1) + ', ' + str(number2))
            else:
                print('Error: ' + submitRsp.data.message)
                time.sleep(1)

        else:
            submitRsp = client.request(
                app.op['Submit'](
                    uid=userId,
                    rid=state.roundId,
                    n1=str(number1)
                ))
            if submitRsp.status == 200:
                print('You submit number: ' + str(number1))
            else:
                print('Error: ' + submitRsp.data.message)
                time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--room', type=int, help='Room ID', required=False)
    args = parser.parse_args()

    main(args.room)