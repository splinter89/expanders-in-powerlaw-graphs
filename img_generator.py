from scipy.special import zetac
import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import random

IS_FOR_PRESENTATION = False

IMG_DIR = os.path.abspath('../repo/images/generated_presentation/' if IS_FOR_PRESENTATION else '../repo/images/generated/')

IMG_DPI = 200
IMG_W = 700
IMG_H = 525
IMG_FONTSIZE = 13
AXIS_MAX = 0.97
AXIS_COLOR = '#fafafa' if IS_FOR_PRESENTATION else 'white'
LABEL_PAD = 2


def tableau20():
    colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)
    return colors


def figsize():
    return IMG_W / float(IMG_DPI), IMG_H / float(IMG_DPI)


def plot_lines(x, y_functions, axis_left, axis_bottom, x_label, y_label, output):
    y_list = []
    for y_func in y_functions:
        y_list.append(np.array(y_func(x)))
    max_x = max(x)
    max_y = np.max(y_list)

    colors = tableau20()
    fig = plt.figure(figsize=figsize(), dpi=IMG_DPI)
    ax = fig.add_axes([axis_left, axis_bottom, AXIS_MAX - axis_left, AXIS_MAX - axis_bottom])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(x_label, labelpad=LABEL_PAD, fontsize=IMG_FONTSIZE)
    ax.set_ylabel(y_label, labelpad=LABEL_PAD, fontsize=IMG_FONTSIZE)
    ax.set_facecolor(AXIS_COLOR)
    plt.xlim(0, max_x)
    plt.ylim(0, max_y * 1.01)
    plt.xticks(fontsize=IMG_FONTSIZE)
    plt.yticks(fontsize=IMG_FONTSIZE)

    if output == 'zeta.png':
        plt.plot([0, max_x], [1] * 2, dashes=[6, 3], lw=1, color='black', alpha=0.3)
        plt.plot([1] * 2, [0, max_y], dashes=[6, 3], lw=1, color='black', alpha=0.3)
    for k, y in enumerate(y_list):
        plt.plot(x, y, linewidth=2.5, color=colors[k])
        if output == 'power-law.png':
            beta_by_k = [0, 0.2, 1, 1.5, 3]
            y_pos_by_k = [0.9, 0.52, 0.27, 0.17, 0.07]
            plt.text(40, y_pos_by_k[k], r'$\beta=$' + str(beta_by_k[k]), fontsize=IMG_FONTSIZE, color=colors[k])

    plt.savefig(os.path.join(IMG_DIR, output), dpi=IMG_DPI, facecolor=AXIS_COLOR)
    # plt.show()


def plot_graph_with_colormap(g, output):
    cmap = plt.cm.Reds
    fig = plt.figure(figsize=figsize(), dpi=IMG_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')

    pos = nx.spring_layout(g)
    degrees = dict(g.degree())

    nx.draw_networkx_edges(g, pos, ax=ax, width=0.75, alpha=0.5)
    nx.draw_networkx_nodes(g, pos, ax=ax,
                           nodelist=list(degrees.keys()),
                           node_size=80,
                           linewidths=0.5,
                           edgecolors='black',
                           node_color=np.array(list(degrees.values())),
                           cmap=cmap)

    norm = plt.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values()) + 2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, fraction=0.18, pad=0, shrink=0.95)
    cb.ax.tick_params(labelsize=IMG_FONTSIZE - 1)
    cb.set_label('Degree', labelpad=0, fontsize=IMG_FONTSIZE - 1)

    plt.savefig(os.path.join(IMG_DIR, output), dpi=IMG_DPI, facecolor=AXIS_COLOR)
    # plt.show()


def plot_deg_distribution(g, axis_left, axis_bottom, x_label, y_label, output):
    colors = tableau20()
    fig = plt.figure(figsize=figsize(), dpi=IMG_DPI)
    ax = fig.add_axes([axis_left, axis_bottom, AXIS_MAX - axis_left, AXIS_MAX - axis_bottom])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(x_label, labelpad=LABEL_PAD, fontsize=IMG_FONTSIZE)
    ax.set_ylabel(y_label, labelpad=LABEL_PAD - 2, fontsize=IMG_FONTSIZE)
    ax.set_facecolor(AXIS_COLOR)
    plt.xticks(fontsize=IMG_FONTSIZE)
    plt.yticks(fontsize=IMG_FONTSIZE)

    deg_sequence = sorted([d for n, d in g.degree()], reverse=True)
    deg_count = collections.Counter(deg_sequence)
    deg, cnt = zip(*deg_count.items())
    ###
    skip_first = 1
    # skip_last = 2
    deg = list(deg)[skip_first:]
    cnt = list(cnt)[skip_first:]
    # deg = list(deg)[skip_first:-skip_last]
    # cnt = list(cnt)[skip_first:-skip_last]
    # print(deg)
    # print(cnt)
    ###
    ax.bar(deg, cnt, width=1, color=colors[0])
    plt.yscale('log', nonposy='clip')

    plt.savefig(os.path.join(IMG_DIR, output), dpi=IMG_DPI, facecolor=AXIS_COLOR)
    # plt.show()


def zeta(x):
    return zetac(x) + 1


def power_law(beta):
    return lambda x: 1 / pow(x, beta)


def power_law_graph(n, beta):
    pl = power_law(beta)
    c = np.power(n, 0.6)
    w = [int(c * pl(x)) for x in range(n, 0, -1)]   # reverse order to fix overlaps of nodes
    # print(nx.is_graphical(w))
    return nx.configuration_model(w)
    # return nx.generators.degree_seq.expected_degree_graph(w)


if __name__ == '__main__':
    # for reproducibility
    random.seed(42)
    np.random.seed(42)

    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    plot_lines(x=np.arange(1.13, 11.13, 0.1), y_functions=[zeta],
               axis_left=0.13, axis_bottom=0.18,
               x_label=r'$\mathfrak{R}(s)$', y_label=r'$\zeta(s)$',
               output='zeta.png')
    plot_lines(x=np.arange(1, 51, 0.1),
               y_functions=[power_law(0), power_law(0.2), power_law(1), power_law(1.5), power_law(3)],
               axis_left=0.18, axis_bottom=0.17,
               x_label=r'$x$', y_label=r'$x^{-\beta}$',
               output='power-law.png')

    G = power_law_graph(n=200, beta=0.4)
    plot_graph_with_colormap(g=G, output='power-law-graph.png')
    plot_deg_distribution(g=G, axis_left=0.17, axis_bottom=0.18,
                          x_label='Degree', y_label='Count',
                          output='power-law-deg-distribution.png')
