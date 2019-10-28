import numpy as np
import pylab as plt
import reversi
from memory import Memory
from tqdm import tqdm


def plot_win_rate(dqn_player, save_name="./fig/win_rate_plot.pdf"):
    wining_DQN = np.array(dqn_player.record) == 1

    final_win_rate = np.cumsum(wining_DQN)[len(
        np.cumsum(wining_DQN))-1]/len(wining_DQN)

    title = []
    title.append("Win_rate (Battle with random(train))")
    if dqn_player.PER:
        title.append("Final = {}, alpha={}, beta={}".format(
            round(final_win_rate, 4), alpha, beta))
    else:
        title.append("Final = {}".format(round(final_win_rate, 4)))

    plt.ylim(0, 1)
    plt.title("\n".join(title))
    plt.xlabel("episode")
    plt.ylabel("win_rate")

    plt.plot(np.cumsum(wining_DQN) / (np.arange(len(wining_DQN)) + 1))
    plt.savefig(save_name, bbox_inches="tight")
    plt.show()


def test_and_plot_win_rate(dqn_player, save_name="./fig/test_plot.pdf", game_num=1000, tqdm=False):
    # train用情報退避
    train_record = dqn_player.record
    dqn_player.record = []

    train_eps = dqn_player.eps
    dqn_player.eps = 0

    train_memory = dqn_player.memory
    dqn_player.memory = Memory(size=64)

    random_player = reversi.player.RandomPlayer("乱太郎")

    if tqdm:
        for i in tqdm(range(game_num)):
            if np.random.random() < 0.5:
                white = random_player
                black = dqn_player
            else:
                white = dqn_player
                black = random_player
            game = reversi.Reversi(white, black)
            game.main_loop(print_game=False)
    else:
        print("battle with random...")
        for i in range(game_num):
            if np.random.random() < 0.5:
                white = random_player
                black = dqn_player
            else:
                white = dqn_player
                black = random_player
            game = reversi.Reversi(white, black)
            game.main_loop(print_game=False)

    wining_DQN = np.array(dqn_player.record) == 1

    plt.grid(True)
    plt.ylim(0, 1)
    final_win_rate = np.cumsum(wining_DQN)[len(
        np.cumsum(wining_DQN))-1]/len(wining_DQN)

    title = []
    title.append("Win_rate (Battle with random(test))")
    if dqn_player.PER:
        title.append("Final = {}, alpha={}, beta={}".format(
            round(final_win_rate, 4), alpha, beta))
    else:
        title.append("Final = {}".format(round(final_win_rate, 4)))

    plt.ylim(0, 1)
    plt.title("\n".join(title))
    plt.xlabel("episode")
    plt.ylabel("win_rate")
    plt.plot(np.cumsum(wining_DQN) / (np.arange(len(wining_DQN)) + 1))
    plt.savefig(save_name, bbox_inches="tight")
    plt.show()

    # train用情報復帰
    dqn_player.record = train_record
    dqn_player.eps = train_eps
    dqn_player.memory = train_memory


def plot_dist_sample(ind_log, bins=640, scale=4):
    plt.figure(figsize=(6*scale, 4*scale))
    plt.hist(ind_log, bins=bins)
    plt.show()


def plot_loss(loss_log, scale=4):
    plt.figure(figsize=(6*scale, 4*scale))
    plt.title("loss")
    plt.ylim(0, 0.5)
    plt.plot(np.arange(len(loss_log)), loss_log, ".")
    plt.show()
