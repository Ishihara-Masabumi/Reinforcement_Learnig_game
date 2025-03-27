import math
import os
import random
import sys

import cv2
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

# ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
from pygame.locals import *

# PygameとPymunkの初期化
pygame.init()
clock = pygame.time.Clock()

# 画面サイズの設定
WIDTH = 800
HEIGHT = 600

# 画面の作成
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reinforcement Learning with Pymunk")

# 色の定義（アルファチャネル付き）
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RED   = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)

# 空間の作成
space = pymunk.Space()
space.gravity = (0.0, 900.0)  # 重力を下向きに設定

draw_options = pymunk.pygame_util.DrawOptions(screen)

# 衝突タイプの定義
COLLISION_TYPE_SQUARE = 1
COLLISION_TYPE_SLOPE = 2
COLLISION_TYPE_DISK = 3

# スロープの定義
def add_slope(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (0, HEIGHT)
    slope_shape = pymunk.Segment(body, (0, 0), (WIDTH, -HEIGHT * 0.1), 1)
    slope_shape.friction = 0.3  # 摩擦係数を減少
    slope_shape.color = WHITE
    slope_shape.collision_type = COLLISION_TYPE_SLOPE  # 衝突タイプを設定
    space.add(body, slope_shape)
    return slope_shape

# スロープの角度を計算
def calculate_slope_angle():
    delta_x = WIDTH
    delta_y = -80
    slope_angle = math.atan2(delta_y, delta_x)
    return slope_angle

# 緑色のディスクを追加
def add_disk(space):
    mass = 1
    radius = 15
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, inertia)
    start_x = random.uniform(WIDTH * 0.7, WIDTH - radius)
    body.position = (start_x, radius)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.5
    shape.friction = 0.5
    shape.color = GREEN
    shape.collision_type = COLLISION_TYPE_DISK  # 衝突タイプを設定
    space.add(body, shape)
    return shape

# 赤い四角形を追加
def add_square(space, slope_angle):
    mass = 5
    size = (30, 30)
    inertia = pymunk.moment_for_box(mass, size)  # 慣性モーメントを計算
    body = pymunk.Body(mass, inertia)
    body.position = (size[0] / 2 + 10, HEIGHT - size[1])
    shape = pymunk.Poly.create_box(body, size)
    shape.friction = 0.3  # 摩擦係数を減少
    shape.elasticity = 0.0
    shape.color = RED
    shape.collision_type = COLLISION_TYPE_SQUARE  # 衝突タイプを設定
    space.add(body, shape)
    return shape

# 環境クラス
class Environment:
    def __init__(self, render=False, save_video=False):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)
        self.slope = add_slope(self.space)
        self.slope_angle = calculate_slope_angle()
        self.square = add_square(self.space, self.slope_angle)
        self.disks = []
        self.done = False
        self.timestep = 0
        self.render_mode = render
        self.save_video = save_video
        if self.save_video:
            self.frames = []
        self.disk_timer = 0  # ディスク生成のタイマー
        self.current_section = int(self.square.body.position.x / (WIDTH / 100))
        self.jump_start_x = None  # ジャンプ開始位置
        self.is_jumping = False  # ジャンプ中かどうか
        self.termination_reason = 0

        # 衝突ハンドラの設定
        handler = self.space.add_collision_handler(COLLISION_TYPE_SQUARE, COLLISION_TYPE_SLOPE)
        handler.begin = self.handle_collision_begin

    def handle_collision_begin(self, arbiter, space, data):
        # 四角形とスロープが接触したとき
        if self.is_jumping:
            self.is_jumping = False  # フラグをリセット
        return True  # 衝突処理を続行

    def reset(self):
        # すべてのオブジェクトを空間から削除
        self.space.remove(self.square.body, self.square)
        for disk in self.disks:
            self.space.remove(disk.body, disk)
        self.disks = []
        self.square = add_square(self.space, self.slope_angle)
        self.done = False
        self.timestep = 0
        self.disk_timer = 0
        self.current_section = int(self.square.body.position.x / (WIDTH / 100))
        self.jump_start_x = None
        self.is_jumping = False
        if self.save_video:
            self.frames = []
        self.generate_disk() # 初期化時にディスクを生成
        return self.get_state()

    def generate_disk(self):
        new_disk = add_disk(self.space)
        self.disks.append(new_disk)

    def get_state(self):
        # 現在のフレームをレンダリングして画像を取得
        screen.fill(BLACK)
        self.space.debug_draw(draw_options)
        pygame.display.flip()

        # 画面を画像としてキャプチャ
        image = pygame.surfarray.array3d(screen)
        image = cv2.transpose(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # カラースレッショルドを使用した物体検出（複数のディスク）
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        disk_info = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                disk_info.append(cX)
                disk_info.append(cY)
                # IDに基づく速度の取得
                matched_disk = None
                min_distance = float('inf')
                for disk in self.disks:
                    disk_x, disk_y = disk.body.position
                    distance = math.hypot(cX - disk_x, cY - disk_y)
                    if distance < min_distance and distance < 50:  # マッチングの閾値
                        min_distance = distance
                        matched_disk = disk
                if matched_disk:
                    vx, vy = matched_disk.body.velocity
                else:
                    vx, vy = 0, 0  # デフォルト値
                disk_info.append(vx)
                disk_info.append(vy)

        # 固定サイズになるようにパディングまたは切り捨て（最大5個のディスク）
        max_disks = 5
        while len(disk_info) < max_disks * 4:
            disk_info.extend([-1000, 1000, 0, 0])  # デフォルト値
        disk_info = disk_info[:max_disks * 4]

        # 四角形の位置を取得
        square_pos = self.square.body.position

        # 状態ベクトル: [ディスク位置と速度（20個の値）, square_x, square_y]
        state = np.array(disk_info + [square_pos.x, square_pos.y], dtype=np.float32)

        return state

    def step(self, action):
        # アクションの適用
        if not self.is_jumping:
            if action == 0:  # 左に移動
                # 速度を直接設定
                move_speed = 1000  # 一定の移動速度
                vx = -move_speed * math.cos(self.slope_angle)
                vy = -move_speed * math.sin(self.slope_angle)
                self.square.body.velocity = (vx, vy)
            elif action == 1:  # 右に移動
                # 速度を直接設定
                move_speed = 1000  # 一定の移動速度
                vx = move_speed * math.cos(self.slope_angle)
                vy = move_speed * math.sin(self.slope_angle)
                self.square.body.velocity = (vx, vy)
            elif action == 2:  # ジャンプ
                if not self.is_jumping:
                    # ジャンプ開始位置を記録
                    self.jump_start_x = self.square.body.position.x
                    self.is_jumping = True
                    # ジャンプ中の横移動を防ぐために速度をゼロに
                    self.square.body.velocity = (0, 0)
                    # 垂直速度を一定値に設定（一定のジャンプ高さ）
                    jump_velocity = -500  # マイナス値で上方向へ（重力が下向きなので）
                    self.square.body.velocity = (0, jump_velocity)
            elif action == 3:  # 何もしない
                # 速度をゼロに設定
                self.square.body.velocity = (0, 0)
        else:
            # ジャンプ中は四角形の位置をジャンプ開始位置に固定
            self.square.body.position = (self.jump_start_x, self.square.body.position.y)
            # 横方向の速度をゼロに
            self.square.body.velocity = (0, self.square.body.velocity.y)

        # 四角形の角速度をゼロに固定
        self.square.body.angular_velocity = 0

        # 物理演算の更新
        dt = 1.0 / 100.0
        self.space.step(dt)

        # ディスクをランダムに生成
        self.disk_timer += 1
        if self.disk_timer > random.randint(120, 360):
            new_disk = add_disk(self.space)
            self.disks.append(new_disk)
            self.disk_timer = 0

        # 左端に到達したディスクを削除
        for disk in self.disks[:]:
            if disk.body.position.x < 0:
                self.space.remove(disk.body, disk)
                self.disks.remove(disk)

        # 報酬の初期化
        reward = -0.1  # 時間ペナルティを減少

        # 衝突のチェック
        collision = False
        for disk in self.disks:
            if self.square.shapes_collide(disk).points:
                self.done = True
                reward -= 100  # 衝突ペナルティ
                self.termination_reason = 3
                collision = True
                break

        # 四角形が右端に到達した場合
        if self.square.body.position.x >= WIDTH:
            self.done = True
            reward += 100  # 成功報酬
            self.termination_reason = 1

        # 四角形が左端に到達した場合
        if self.square.body.position.x <= 0:
            self.done = True
            reward -= 100  # 失敗ペナルティ
            self.termination_reason = 2

        # アクションによる報酬
        if action == 0:  # 左に移動
            reward -= 1
        elif action == 1:  # 右に移動
            reward += 1
        elif action == 2:  # ジャンプ
            reward -= 1

        # セクションの更新
        section_width = WIDTH / 100
        new_section = int(self.square.body.position.x / section_width)
        new_section = max(0, min(99, new_section))

        # 移動による報酬
        if new_section != self.current_section:
            if new_section > self.current_section:
                reward += new_section - self.current_section  # 右に移動した場合
            elif new_section < self.current_section:
                reward -= self.current_section - new_section   # 左に移動した場合
            self.current_section = new_section

        if not self.is_jumping:
            self.timestep += 1
            if self.timestep > 1000:
                self.done = True  # タイムリミット

        state = self.get_state()

        if self.render_mode:
            self.render()

        if self.save_video:
            frame = pygame.surfarray.array3d(screen)
            frame = cv2.transpose(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.frames.append(frame)

        return state, reward, self.done, self.termination_reason, self.timestep

    def render(self):
        screen.fill(BLACK)
        self.space.debug_draw(draw_options)
        pygame.display.flip()
        clock.tick(60)

    def save_video_to_file(self, filename):
        if self.save_video and self.frames:
            height, width, _ = self.frames[0].shape
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()

# DQNエージェントの定義（変更なし）
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# 優先度付き経験再生メモリ（変更なし）
class PrioritizedReplayMemory:
    def __init__(self, capacity=5000, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.alpha = alpha
        self.position = 0

    def push(self, transition):
        max_priority = max(self.priorities) if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = transition
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probs = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=sample_probs)
        samples = [self.memory[i] for i in indices]

        # 重要度サンプリングの重みを計算
        total = len(self.memory)
        weights = (total * sample_probs[indices]) ** (-beta)
        weights = weights / weights.max()

        # weights が numpy.ndarray ならば torch.Tensor に変換
        if isinstance(weights, np.ndarray):
            weights = torch.tensor(weights, dtype=torch.float32)

        # PyTorchのテンソルとして操作 
        weights = weights.clone().detach().float()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error.item()) + 1e-5  # ゼロ除算を防ぐために微小値を加算

    def __len__(self):
        return len(self.memory)

# トレーニングループ（変更なし）
def train_agent(policy):

    num_episodes = 1000
    batch_size = 124
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 5000
    target_update = 10

    beta_start = 0.4
    beta_frames = num_episodes * 1000

    state_size = 22  # 5ディスク * 2（x, y）+ square_x + square_y
    action_size = 4

    if policy == 1:

        env = Environment()

        policy_net = DQNAgent(state_size, action_size)
        target_net = DQNAgent(state_size, action_size)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    else:
        env = Environment()

        policy_net = DQNAgent(state_size, action_size)
        target_net = DQNAgent(state_size, action_size)
        policy_net.load_state_dict(torch.load('dqn_model.pth'))
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        # rewards.npy のデータを読み込み
        rewards = np.load('rewards.npy')

        # エピソード数を数える
        num_epochs = len(rewards)
        print(f"開始エピソード数: {num_epochs}")

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = PrioritizedReplayMemory(10000)

    steps_done = 0
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0
        while True:
            if not env.is_jumping:
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                          np.exp(-1. * steps_done / epsilon_decay)
                steps_done += 1

                if random.random() < epsilon:
                    action = random.randrange(action_size)
                else:
                    with torch.no_grad():
                        state_tensor = torch.from_numpy(state)
                        q_values = policy_net(state_tensor)
                        action = q_values.argmax().item()

                next_state, reward, done, reason, step = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                total_reward += reward
                #print("action=: " + str(action))

                # 経験をリプレイメモリに保存
                memory.push((state, action, reward, next_state, done))

                if len(memory) > batch_size:
                    beta = min(1.0, beta_start + steps_done * (1.0 - beta_start) / beta_frames)
                    samples, indices, weights = memory.sample(batch_size, beta)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*samples)

                    batch_state = np.array(batch_state, dtype=np.float32)
                    batch_next_state = np.array(batch_next_state, dtype=np.float32)

                    batch_state = torch.tensor(batch_state)
                    batch_action = torch.tensor(batch_action, dtype=torch.long)
                    batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
                    batch_next_state = torch.tensor(batch_next_state)
                    batch_done = torch.tensor(batch_done, dtype=torch.bool)

                    # weights が numpy.ndarray ならば torch.Tensor に変換
                    if isinstance(weights, np.ndarray):
                        weights = torch.tensor(weights, dtype=torch.float32)

                    # PyTorchのテンソルとして操作 
                        weights = weights.clone().detach().float()

                    current_q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                    next_q_values = target_net(batch_next_state).max(1)[0]
                    next_q_values[batch_done] = 0.0
                    expected_q_values = batch_reward + gamma * next_q_values

                    td_errors = current_q_values - expected_q_values.detach()
                    loss = (weights * td_errors ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 優先度を更新
                    memory.update_priorities(indices, td_errors.detach())

                state = next_state  # 状態を更新

            else:
                # ジャンプ中は推論・学習・データ保存を行わない
                action = 3  # 何もしない
                next_state, reward, done, reason, step = env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                total_reward += reward
                state = next_state  # 状態を更新

            if done:
                print(f"Episode {episode+1}/{num_episodes}, Timestep: {step}, Total Reward: {total_reward}, Terminate: {reason}")
                break

        all_rewards.append(total_reward)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # モデルを保存
    torch.save(policy_net.state_dict(), 'dqn_model.pth')
    # 報酬を保存
    np.save('rewards.npy', np.array(all_rewards))

# 可視化用のスクリプト（変更なし）
def visualize_trial(i):
    env = Environment(render=True, save_video=True)
    state_size = 22
    action_size = 4

    policy_net = DQNAgent(state_size, action_size)
    policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=torch.device('cuda')))
    policy_net.eval()

    state = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0

    while True:
        if not env.is_jumping:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
        else:
            action = 3  # ジャンプ中は何もしない

        next_state, reward, done, reason, step = env.step(action)
        total_reward += reward

        state = np.array(next_state, dtype=np.float32)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        if done:
            break

    env.save_video_to_file(f'last_epoch_video_{i}.mp4')
    print(f"Number {i}/10, Timestep: {step}, Reward: {total_reward}, Terminate: {reason}")

if __name__ == "__main__":

    policy = 1

    # エージェントをトレーニング
    if policy > 0:
        train_agent(policy)

    # 試行を可視化してビデオを保存
    for i in range(10):
        visualize_trial(i)

    # 終了
    pygame.quit()
    sys.exit()



