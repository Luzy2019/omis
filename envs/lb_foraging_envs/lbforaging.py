import argparse
import logging
import random
import time
import gym
import numpy as np
import lbforaging
import cv2

logger = logging.getLogger(__name__)

def gen_video(imgs, filename, size=(1000, 1000)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, 2.5, size)
    for i in imgs:
        Img = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
        Img = cv2.resize(Img, size)
        video_writer.write(Img)
    print(f"Video {filename} writen down!")

def _game_loop(env, render):
    """
    """
    if render:
        imgs = []
    obs = env.reset()
    done = False

    if render:
        img = np.array(env.render(mode='rgb_array'))
        height, width, _ = img.shape
        imgs.append(img)

    while not done:

        actions = env.action_space.sample()

        nobs, nreward, ndone, _ = env.step(actions)
        if sum(nreward) > 0:
            print(nreward)

        if render:
            img = np.array(env.render(mode='rgb_array'))
            imgs.append(img)

        done = np.all(ndone)
    # print(env.players[0].score, env.players[1].score)

    if render:
        gen_video(imgs, "test.mp4", size=(width, height))


def main(game_count=1, render=False):
    env = gym.make("Foraging-9x9-2p-5f-v2")
    obs = env.reset()

    for episode in range(game_count):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
