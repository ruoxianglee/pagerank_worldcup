import cvxpy as cp
import numpy as np
import networkx as nx
from random import randrange
import matplotlib.pyplot as plt
import my_networkx as my_nx

def page_rank_cvx(teleport_prob, P, v, teams):
    # code reference: tutorial 6

    N = len(v)
    pi = cp.Variable(N)
    I = np.identity(N)
    a = 1 - teleport_prob
    objective = cp.Minimize(cp.norm((I - a * P) @ pi - (1 - a) * v, 2))
    constraint = [cp.sum(pi) == 1, pi >= 0]
    problem = cp.Problem(objective, constraint)
    problem.solve()
    print('Solving Status:', problem.status)

    results = []
    if problem.status == 'optimal':
        print('Optimal Variable:', pi.value)
    for i in pi.value.argsort()[::-1]:
        results.append(teams[i])
        print('Country {}: rank {}'.format(teams[i], pi.value[i]))

    print('-' * 20)

    return results

def update(teleport_prob, P, v, pre_pi):
    part1 = P @ pre_pi
    updated_pi = (1 - teleport_prob) * part1 + teleport_prob * v
    return updated_pi

def iteration_method(teleport_prob, P, v, initial_pi, max_itrs, teams):
    tol = 0.000001
    pre_pi = initial_pi
    updated_pi = []
    for i in range(max_itrs):
        updated_pi = update(teleport_prob, P, v, pre_pi)
        for j in range(len(updated_pi)):
            if abs(updated_pi[j]-pre_pi[j]) > tol:
                break
        else:
            break
        pre_pi = updated_pi
    
    team_ranking = {}
    for i in range(len(teams)):
        team_ranking[teams[i]] = updated_pi[i][0]
    team_ranking = sorted(team_ranking.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)

    results = []
    for key, value in team_ranking:
        results.append(key)
        print('Country {}: rank {}'.format(key, value))

    print('-' * 20)

    return results

def plot_graph(G, group):
    plt.figure(figsize=(7.5, 7.5))
    plt.axis('off')

    options = {
        'font_weight': 'bold',
        "with_labels": True,
        "connectionstyle": 'arc3, rad = 0.1',
    }

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)

    # code reference: https://stackoverflow.com/questions/22785849/drawing-multiple-edges-between-two-nodes-with-networkx
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))
    nx.draw_networkx_edges(G, pos, edgelist=straight_edges, **options)
    nx.draw_networkx_edges(G, pos, edgelist=curved_edges, **options)

    weights = nx.get_edge_attributes(G, 'weight')
    curved_edge_labels = {edge: weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: weights[edge] for edge in straight_edges}
    my_nx.my_draw_networkx_edge_labels(G, pos, edge_labels = curved_edge_labels, rotate = False, rad = 0.1)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = straight_edge_labels, rotate = False)

    plt.title(group, fontdict = {'fontsize' : 40})
    plt.savefig(group + ".png")
    plt.close()

def generate_directed_graph(group_dictionary, groups):
    graphs = []
    for i in range(len(group_dictionary)):
        g = groups[group_dictionary[i]]
        directed_Graph = nx.DiGraph()
        nodes = g

        edges = []
        for j in range(len(g) - 1):
            for k in range(len(g) - 1 - j):
                w1 = randrange(20)
                if w1 > 0:
                    edge1 = (g[j], g[k + j + 1], w1)
                    edges.append(edge1)
                
                if w1 == 0:
                    w2 = randrange(1,20)
                    edge2 = (g[k + j + 1], g[j], w2)
                    edges.append(edge2)
                else:
                    w2 = randrange(20)
                    if w2 > 0:
                        edge2 = (g[k + j + 1], g[j], w2)
                        edges.append(edge2)
                
        directed_Graph.add_nodes_from(nodes)
        directed_Graph.add_weighted_edges_from(edges)
        graphs.append(directed_Graph)

        plot_graph(directed_Graph, group_dictionary[i])

    return graphs

def create_stochastic_matrix(graphs):
    
    sto_matrixs = []
    teams_by_group = []
    for graph in graphs:
        weights = dict(nx.get_edge_attributes(graph, 'weight'))

        teams = list(graph.nodes())
        teams_by_group.append(teams)
        out_degree_dic = {}
        for team in teams:
            out_degree_dic[team] =  0

        for key in weights:
            if key[0] in teams:
                out_degree_dic[key[0]] = out_degree_dic[key[0]] + weights[key]
        
        N = len(teams)
        substo_mat = np.empty(shape=(N,0), order='c')
        for i in range(N):
            team = teams[i]
            outlinks = []
            for j in range(N):
                if j == i:
                    outlinks.append(0)
                    continue

                if (team, teams[j]) in weights:
                    wij = weights[(team, teams[j])]
                else:
                    wij = 0

                if out_degree_dic[team] == 0:
                    Pij = 0
                else:
                    Pij = wij / out_degree_dic[team]
                outlinks.append(Pij)

            col = np.array(outlinks).reshape(N,1)
            substo_mat = np.append(substo_mat, col, axis=1)

        # convert sub-stochastic to stochastic with weakly preferential modification
        e = np.array([1] * N).reshape(N, 1)
        c = e.T - e.T @ substo_mat
        u = e / N

        sto_matrixs.append(substo_mat + u*c)

    return sto_matrixs, teams_by_group

def predict_champion_by_cvx(group_dictionary, groups):
    
    # Group Stage
    print('-' * 20, 'Group Stage', '-' * 20)
    
    graphs = generate_directed_graph(group_dictionary, groups)
    sto_matrixs, teams_by_group = create_stochastic_matrix(graphs)

    winner_teams_upper_half = []
    runner_up_teams_upper_half = []
    winner_teams_lower_half = []
    runner_up_teams_lower_half = []

    for i in range(len(sto_matrixs)):
        v = np.array([1/4] * 4)
        result = page_rank_cvx(0.85, sto_matrixs[i], v, teams_by_group[i])

        if i < 4:
            winner_teams_upper_half.append(result[0])
            runner_up_teams_upper_half.append(result[1])
        else:
            winner_teams_lower_half.append(result[0])
            runner_up_teams_lower_half.append(result[1])
    
    # Round of 16
    print('-' * 20, 'Round of 16', '-' * 20)

    group_dictionary = ["r16_1", "r16_2", "r16_3", "r16_4", "r16_5", "r16_6", "r16_7", "r16_8"]

    groups_upper = {}
    groups_lower = {}

    groups_upper[group_dictionary[0]] = [winner_teams_upper_half[0], runner_up_teams_upper_half[1]]
    groups_upper[group_dictionary[1]] = [winner_teams_upper_half[1], runner_up_teams_upper_half[0]]
    groups_upper[group_dictionary[2]] = [winner_teams_upper_half[2], runner_up_teams_upper_half[3]]
    groups_upper[group_dictionary[3]] = [winner_teams_upper_half[3], runner_up_teams_upper_half[2]]

    groups_lower[group_dictionary[4]] = [winner_teams_lower_half[0], runner_up_teams_lower_half[1]]
    groups_lower[group_dictionary[5]] = [winner_teams_lower_half[1], runner_up_teams_lower_half[0]]
    groups_lower[group_dictionary[6]] = [winner_teams_lower_half[2], runner_up_teams_lower_half[3]]
    groups_lower[group_dictionary[7]] = [winner_teams_lower_half[3], runner_up_teams_lower_half[2]]

    graphs_upper_half = generate_directed_graph(group_dictionary[0:4], groups_upper)
    sto_matrixs_upper_half, teams_by_group_upper_half = create_stochastic_matrix(graphs_upper_half)

    graphs_lower_half = generate_directed_graph(group_dictionary[4:8], groups_lower)
    sto_matrixs_lower_half, teams_by_group_lower_half = create_stochastic_matrix(graphs_lower_half)

    winner_teams_upper_half = []
    winner_teams_lower_half = []

    for i in range(len(sto_matrixs_upper_half)):
        v = np.array([1/2] * 2)
        result = page_rank_cvx(0.85, sto_matrixs_upper_half[i], v, teams_by_group_upper_half[i])
        winner_teams_upper_half.append(result[0])

    for i in range(len(sto_matrixs_lower_half)):
        v = np.array([1/2] * 2)
        result = page_rank_cvx(0.85, sto_matrixs_lower_half[i], v, teams_by_group_lower_half[i])
        winner_teams_lower_half.append(result[0])

    # quarter-final
    print('-' * 20, 'Quarter-finals', '-' * 20)

    group_dictionary = ["quarter_1", "quarter_2", "quarter_3", "quarter_4"]

    groups_upper = {}
    groups_lower = {}

    groups_upper[group_dictionary[0]] = [winner_teams_upper_half[0], winner_teams_upper_half[2]]
    groups_upper[group_dictionary[1]] = [winner_teams_upper_half[1], winner_teams_upper_half[3]]

    groups_lower[group_dictionary[2]] = [winner_teams_lower_half[0], winner_teams_lower_half[2]]
    groups_lower[group_dictionary[3]] = [winner_teams_lower_half[1], winner_teams_lower_half[3]]

    graphs_upper_half = generate_directed_graph(group_dictionary[0:2], groups_upper)
    sto_matrixs_upper_half, teams_by_group_upper_half = create_stochastic_matrix(graphs_upper_half)

    graphs_lower_half = generate_directed_graph(group_dictionary[2:4], groups_lower)
    sto_matrixs_lower_half, teams_by_group_lower_half = create_stochastic_matrix(graphs_lower_half)

    winner_teams_upper_half = []
    winner_teams_lower_half = []

    for i in range(len(sto_matrixs_upper_half)):
        v = np.array([1/2] * 2)
        result = page_rank_cvx(0.85, sto_matrixs_upper_half[i], v, teams_by_group_upper_half[i])
        winner_teams_upper_half.append(result[0])

    for i in range(len(sto_matrixs_lower_half)):
        v = np.array([1/2] * 2)
        result = page_rank_cvx(0.85, sto_matrixs_lower_half[i], v, teams_by_group_lower_half[i])
        winner_teams_lower_half.append(result[0])

    # semi-final
    print('-' * 20, 'Semi-finals', '-' * 20)

    group_dictionary = ["semifinal_1", "semifinal_2"]

    groups_upper = {}
    groups_lower = {}

    groups_upper[group_dictionary[0]] = [winner_teams_upper_half[0], winner_teams_upper_half[1]]

    groups_lower[group_dictionary[1]] = [winner_teams_lower_half[0], winner_teams_lower_half[1]]

    graphs_upper_half = generate_directed_graph([group_dictionary[0]], groups_upper)
    sto_matrixs_upper_half, teams_by_group_upper_half = create_stochastic_matrix(graphs_upper_half)

    graphs_lower_half = generate_directed_graph([group_dictionary[1]], groups_lower)
    sto_matrixs_lower_half, teams_by_group_lower_half = create_stochastic_matrix(graphs_lower_half)

    winner_teams_upper_half = []
    winner_teams_lower_half = []

    for i in range(len(sto_matrixs_upper_half)):
        v = np.array([1/2] * 2)
        result = page_rank_cvx(0.85, sto_matrixs_upper_half[i], v, teams_by_group_upper_half[i])
        winner_teams_upper_half.append(result[0])

    for i in range(len(sto_matrixs_lower_half)):
        v = np.array([1/2] * 2)
        result = page_rank_cvx(0.85, sto_matrixs_lower_half[i], v, teams_by_group_lower_half[i])
        winner_teams_lower_half.append(result[0])

    # final
    print('-' * 20, 'Finals', '-' * 20)

    group_dictionary = ["final"]

    group = {}

    group[group_dictionary[0]] = [winner_teams_upper_half[0], winner_teams_lower_half[0]]

    graph = generate_directed_graph(group_dictionary, group)
    sto_matrix, teams_by_group = create_stochastic_matrix(graph)

    v = np.array([1/2] * 2)
    result = page_rank_cvx(0.85, sto_matrix[0], v, teams_by_group[0])

    print("Champion Team: {} ! Congratulations!".format(result[0]))

def predict_champion_by_iteration(group_dictionary, groups):
   
    # Group Stage
    print('-' * 20, 'Group Stage', '-' * 20)
    
    graphs = generate_directed_graph(group_dictionary, groups)
    sto_matrixs, teams_by_group = create_stochastic_matrix(graphs)

    winner_teams_upper_half = []
    runner_up_teams_upper_half = []
    winner_teams_lower_half = []
    runner_up_teams_lower_half = []

    for i in range(len(sto_matrixs)):
        initial_pi = np.array([1/4] * 4).reshape(4, 1)
        v = np.array([1/4] * 4).reshape(4,1)
        result = iteration_method(0.85, sto_matrixs[i], v, initial_pi, 1000, teams_by_group[i])

        if i < 4:
            winner_teams_upper_half.append(result[0])
            runner_up_teams_upper_half.append(result[1])
        else:
            winner_teams_lower_half.append(result[0])
            runner_up_teams_lower_half.append(result[1])
    
    # Round of 16
    print('-' * 20, 'Round of 16', '-' * 20)

    group_dictionary = ["r16_1", "r16_2", "r16_3", "r16_4", "r16_5", "r16_6", "r16_7", "r16_8"]

    groups_upper = {}
    groups_lower = {}

    groups_upper[group_dictionary[0]] = [winner_teams_upper_half[0], runner_up_teams_upper_half[1]]
    groups_upper[group_dictionary[1]] = [winner_teams_upper_half[1], runner_up_teams_upper_half[0]]
    groups_upper[group_dictionary[2]] = [winner_teams_upper_half[2], runner_up_teams_upper_half[3]]
    groups_upper[group_dictionary[3]] = [winner_teams_upper_half[3], runner_up_teams_upper_half[2]]

    groups_lower[group_dictionary[4]] = [winner_teams_lower_half[0], runner_up_teams_lower_half[1]]
    groups_lower[group_dictionary[5]] = [winner_teams_lower_half[1], runner_up_teams_lower_half[0]]
    groups_lower[group_dictionary[6]] = [winner_teams_lower_half[2], runner_up_teams_lower_half[3]]
    groups_lower[group_dictionary[7]] = [winner_teams_lower_half[3], runner_up_teams_lower_half[2]]

    graphs_upper_half = generate_directed_graph(group_dictionary[0:4], groups_upper)
    sto_matrixs_upper_half, teams_by_group_upper_half = create_stochastic_matrix(graphs_upper_half)

    graphs_lower_half = generate_directed_graph(group_dictionary[4:8], groups_lower)
    sto_matrixs_lower_half, teams_by_group_lower_half = create_stochastic_matrix(graphs_lower_half)

    winner_teams_upper_half = []
    winner_teams_lower_half = []

    for i in range(len(sto_matrixs_upper_half)):
        initial_pi = np.array([1/2] * 2).reshape(2, 1)
        v = np.array([1/2] * 2).reshape(2,1)
        result = iteration_method(0.85, sto_matrixs_upper_half[i], v, initial_pi, 1000, teams_by_group_upper_half[i])

        winner_teams_upper_half.append(result[0])

    for i in range(len(sto_matrixs_lower_half)):
        initial_pi = np.array([1/2] * 2).reshape(2, 1)
        v = np.array([1/2] * 2).reshape(2,1)
        result = iteration_method(0.85, sto_matrixs_lower_half[i], v, initial_pi, 1000, teams_by_group_lower_half[i])
        winner_teams_lower_half.append(result[0])

    # quarter-final
    print('-' * 20, 'Quarter-finals', '-' * 20)

    group_dictionary = ["quarter_1", "quarter_2", "quarter_3", "quarter_4"]

    groups_upper = {}
    groups_lower = {}

    groups_upper[group_dictionary[0]] = [winner_teams_upper_half[0], winner_teams_upper_half[2]]
    groups_upper[group_dictionary[1]] = [winner_teams_upper_half[1], winner_teams_upper_half[3]]

    groups_lower[group_dictionary[2]] = [winner_teams_lower_half[0], winner_teams_lower_half[2]]
    groups_lower[group_dictionary[3]] = [winner_teams_lower_half[1], winner_teams_lower_half[3]]

    graphs_upper_half = generate_directed_graph(group_dictionary[0:2], groups_upper)
    sto_matrixs_upper_half, teams_by_group_upper_half = create_stochastic_matrix(graphs_upper_half)

    graphs_lower_half = generate_directed_graph(group_dictionary[2:4], groups_lower)
    sto_matrixs_lower_half, teams_by_group_lower_half = create_stochastic_matrix(graphs_lower_half)

    winner_teams_upper_half = []
    winner_teams_lower_half = []

    for i in range(len(sto_matrixs_upper_half)):
        initial_pi = np.array([1/2] * 2).reshape(2, 1)
        v = np.array([1/2] * 2).reshape(2,1)
        result = iteration_method(0.85, sto_matrixs_upper_half[i], v, initial_pi, 1000, teams_by_group_upper_half[i])
        
        winner_teams_upper_half.append(result[0])

    for i in range(len(sto_matrixs_lower_half)):
        initial_pi = np.array([1/2] * 2).reshape(2, 1)
        v = np.array([1/2] * 2).reshape(2,1)
        result = iteration_method(0.85, sto_matrixs_lower_half[i], v, initial_pi, 1000, teams_by_group_lower_half[i])
        winner_teams_lower_half.append(result[0])

    # semi-final
    print('-' * 20, 'Semi-finals', '-' * 20)

    group_dictionary = ["semifinal_1", "semifinal_2"]

    groups_upper = {}
    groups_lower = {}

    groups_upper[group_dictionary[0]] = [winner_teams_upper_half[0], winner_teams_upper_half[1]]

    groups_lower[group_dictionary[1]] = [winner_teams_lower_half[0], winner_teams_lower_half[1]]

    graphs_upper_half = generate_directed_graph([group_dictionary[0]], groups_upper)
    sto_matrixs_upper_half, teams_by_group_upper_half = create_stochastic_matrix(graphs_upper_half)

    graphs_lower_half = generate_directed_graph([group_dictionary[1]], groups_lower)
    sto_matrixs_lower_half, teams_by_group_lower_half = create_stochastic_matrix(graphs_lower_half)

    winner_teams_upper_half = []
    winner_teams_lower_half = []

    for i in range(len(sto_matrixs_upper_half)):
        initial_pi = np.array([1/2] * 2).reshape(2, 1)
        v = np.array([1/2] * 2).reshape(2,1)
        result = iteration_method(0.85, sto_matrixs_upper_half[i], v, initial_pi, 1000, teams_by_group_upper_half[i])

        winner_teams_upper_half.append(result[0])

    for i in range(len(sto_matrixs_lower_half)):
        initial_pi = np.array([1/2] * 2).reshape(2, 1)
        v = np.array([1/2] * 2).reshape(2,1)
        result = iteration_method(0.85, sto_matrixs_lower_half[i], v, initial_pi, 1000, teams_by_group_lower_half[i])
        winner_teams_lower_half.append(result[0])

    # final
    print('-' * 20, 'Finals', '-' * 20)

    group_dictionary = ["final"]

    group = {}

    group[group_dictionary[0]] = [winner_teams_upper_half[0], winner_teams_lower_half[0]]

    graph = generate_directed_graph(group_dictionary, group)
    sto_matrix, teams_by_group = create_stochastic_matrix(graph)

    initial_pi = np.array([1/2] * 2).reshape(2, 1)
    v = np.array([1/2] * 2).reshape(2,1)
    result = iteration_method(0.85, sto_matrix[0], v, initial_pi, 1000, teams_by_group[0])

    print("Champion Team: {} ! Congratulations!".format(result[0]))

if "__main__" == __name__:

    group_dictionary = ["groupA_teams", "groupB_teams", "groupC_teams", "groupD_teams", "groupE_teams", "groupF_teams", "groupG_teams", "groupH_teams"]

    groups = {  "groupA_teams": [ "Qatar", "Ecuador", "Senegal", "Netherlands" ],
                "groupB_teams": [ "England", "Iran", "USA", "Wales" ],
                "groupC_teams": [ "Argentina", "SaudiArabia", "Mexico", "Poland" ],
                "groupD_teams": [ "France", "Australia", "Denmark", "Tunisia" ],
                "groupE_teams": [ "Spain", "NewZealand", "Germany", "Japan" ],
                "groupF_teams": [ "Belgium", "Canada", "Morocco", "Croatia" ],
                "groupG_teams": [ "Brazil", "Serbia", "Switzerland", "Cameroon" ],
                "groupH_teams": [ "Portugal", "Ghana", "Uruguay", "Korea"]
    }
  
    # Predict champion using cvxpy
    # predict_champion_by_cvx(group_dictionary, groups)

    # Predict champion using iteration method
    predict_champion_by_iteration(group_dictionary, groups)