import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import random
from collections import defaultdict

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False  

# Clear all variables and graphics
plt.close('all')

try:
    with open('data/team_mapping.json', 'r', encoding='utf-8') as f:
        team_mapping = json.load(f)

    train_data = pd.read_csv('data/processed_train.csv')
    test_data = pd.read_csv('data/processed_test.csv')
except FileNotFoundError as e:
    print(f"Error: Missing required input file: {e.filename}")
    print("Please run 'python preprocess.py' to generate the required data files.")
    exit()

id_to_team = {}
for team_id, team_name in team_mapping['id_to_team'].items():
    clean_name = team_name.replace('市', '')  
    id_to_team[int(team_id)] = clean_name

# Create team list (sorted by ID)
teams = [id_to_team[i] for i in sorted(id_to_team.keys())]
n_teams = len(teams)

print(f"Teams list: {teams}")

print(f"Historical match data: {len(train_data)} matches")
print(f"Matches to predict: {len(test_data)} matches")

# Elo rating system parameters
base_elo = 1500
k_factor = 32  # Elo update factor
home_advantage = 100  # Home advantage

current_elo = np.full(n_teams, base_elo, dtype=float)
current_points = np.zeros(n_teams)
games_played = np.zeros(n_teams)
wins = np.zeros(n_teams)
draws = np.zeros(n_teams)
losses = np.zeros(n_teams)

print("=== Update Elo rating and points based on historical games ===")

for _, match in train_data.iterrows():
    home_id = int(match['home_team_id'])
    away_id = int(match['away_team_id'])
    home_score = match['home_score']
    away_score = match['away_score']
    
    games_played[home_id] += 1
    games_played[away_id] += 1
    
    if home_score > away_score:
        current_points[home_id] += 3
        wins[home_id] += 1
        losses[away_id] += 1
        actual_score = 1.0  
    elif home_score < away_score:
        current_points[away_id] += 3
        wins[away_id] += 1
        losses[home_id] += 1
        actual_score = 0.0  
    else:
        current_points[home_id] += 1
        current_points[away_id] += 1
        draws[home_id] += 1
        draws[away_id] += 1
        actual_score = 0.5  
    
    # Get current Elo rating (home team has home advantage)
    home_elo = current_elo[home_id] + home_advantage
    away_elo = current_elo[away_id]
    
    # Calculate the expected win rate
    expected_home_win = 1 / (1 + 10**((away_elo - home_elo) / 400))
    
    # Update Elo rating
    elo_change = k_factor * (actual_score - expected_home_win)
    current_elo[home_id] += elo_change
    current_elo[away_id] -= elo_change

print("=== 当前联赛积分榜 ===")
print("排名\t球队\t积分\t场次\t胜\t平\t负\tElo评分")
# 按积分排序
points_ranking = np.argsort(current_points)[::-1]
for i, team_id in enumerate(points_ranking):
    team_name = teams[team_id]
    points = current_points[team_id]
    played = int(games_played[team_id])
    w = int(wins[team_id])
    d = int(draws[team_id])
    l = int(losses[team_id])
    elo = current_elo[team_id]
    print(f"{i+1}\t{team_name}\t{points:.0f}\t{played}\t{w}\t{d}\t{l}\t{elo:.1f}")

print("\n=== Predicting remaining match results ===")
remaining_games = 12 - games_played
predicted_additional_points = np.zeros(n_teams)

# Simulate remaining games (based on test data)
for _, match in test_data.iterrows():
    home_id = int(match['home_team_id'])
    away_id = int(match['away_team_id'])
    
    home_elo = current_elo[home_id] + home_advantage
    away_elo = current_elo[away_id]
    
    home_win_prob = 1 / (1 + 10**((away_elo - home_elo) / 400))
    # Set draw probability based on Elo difference
    elo_diff = abs(home_elo - away_elo)
    draw_prob = max(0.1, 0.3 - elo_diff / 4000)
    away_win_prob = 1 - home_win_prob - draw_prob
        # 调整概率确保和为1
    total_prob = home_win_prob + away_win_prob
    home_win_prob_adj = home_win_prob / total_prob * (1 - draw_prob)
    away_win_prob_adj = away_win_prob / total_prob * (1 - draw_prob)
    
    expected_home_points = 3 * home_win_prob_adj + 1 * draw_prob
    expected_away_points = 3 * away_win_prob_adj + 1 * draw_prob
    
    predicted_additional_points[home_id] += expected_home_points
    predicted_additional_points[away_id] += expected_away_points

predicted_final_points = current_points + predicted_additional_points

print("\n=== Regular season final points prediction ===")
print("排名\t球队\t当前积分\t预测增加\t预测最终积分")
final_ranking = np.argsort(predicted_final_points)[::-1]
for i, team_id in enumerate(final_ranking):
    team_name = teams[team_id]
    current = current_points[team_id]
    additional = predicted_additional_points[team_id]
    final = predicted_final_points[team_id]
    print(f"{i+1}\t{team_name}\t{current:.0f}\t\t{additional:.1f}\t\t{final:.1f}")

# Monte Carlo simulation for playoffs
n_simulations = 10000
champions = np.zeros(n_simulations, dtype=int)
finalists = np.zeros((2, n_simulations), dtype=int)
semifinalists = np.zeros((4, n_simulations), dtype=int)
playoff_qualifiers = np.zeros((8, n_simulations), dtype=int)

print(f'\nRunning {n_simulations} complete simulations (regular season + playoffs)...')

random.seed(42)
np.random.seed(42)

for sim in range(n_simulations):
    sim_elo = current_elo.copy()
    sim_points = current_points.copy()
    sim_games_played = games_played.copy()
    
    for _, match in test_data.iterrows():
        home_id = int(match['home_team_id'])
        away_id = int(match['away_team_id'])
        
        home_elo = sim_elo[home_id] + home_advantage
        away_elo = sim_elo[away_id]
        
        home_win_prob = 1 / (1 + 10**((away_elo - home_elo) / 400))
        elo_diff = abs(home_elo - away_elo)
        draw_prob = max(0.1, 0.3 - elo_diff / 4000)
        away_win_prob = 1 - home_win_prob - draw_prob
        
        total_prob = home_win_prob + away_win_prob
        home_win_prob_adj = home_win_prob / total_prob * (1 - draw_prob)
        away_win_prob_adj = away_win_prob / total_prob * (1 - draw_prob)
        
        rand_val = random.random()
        if rand_val < home_win_prob_adj:
            sim_points[home_id] += 3
            actual_score = 1.0
        elif rand_val < home_win_prob_adj + draw_prob:
            sim_points[home_id] += 1
            sim_points[away_id] += 1
            actual_score = 0.5
        else:
            sim_points[away_id] += 3
            actual_score = 0.0
        
        expected_home_win = 1 / (1 + 10**((away_elo - home_elo) / 400))
        elo_change = k_factor * (actual_score - expected_home_win)
        sim_elo[home_id] += elo_change
        sim_elo[away_id] -= elo_change
        
        sim_games_played[home_id] += 1
        sim_games_played[away_id] += 1
    
    sim_final_ranking = np.argsort(sim_points)[::-1]
    sim_playoff_teams = sim_final_ranking[:8]
    playoff_qualifiers[:, sim] = sim_playoff_teams
    
    qf_pairings = [[0, 7], [1, 6], [2, 5], [3, 4]]
    qf_winners = []
    
    for i in range(4):
        team1_idx = sim_playoff_teams[qf_pairings[i][0]]
        team2_idx = sim_playoff_teams[qf_pairings[i][1]]
        
        elo1 = sim_elo[team1_idx] + home_advantage
        elo2 = sim_elo[team2_idx]
        
        win_prob1 = 1 / (1 + 10**((elo2 - elo1) / 400))
        
        if random.random() < win_prob1:
            qf_winners.append(team1_idx)
        else:
            qf_winners.append(team2_idx)
    
    # Semi-final pairings
    sf_pairings = [[0, 3], [1, 2]]
    sf_winners = []
    sf_losers = []
    
    for i in range(2):
        team1_idx = qf_winners[sf_pairings[i][0]]
        team2_idx = qf_winners[sf_pairings[i][1]]
        
        team1_rank = np.where(sim_playoff_teams == team1_idx)[0][0]
        team2_rank = np.where(sim_playoff_teams == team2_idx)[0][0]
        
        if team1_rank < team2_rank:
            elo1 = sim_elo[team1_idx] + home_advantage
            elo2 = sim_elo[team2_idx]
        else:
            elo1 = sim_elo[team1_idx]
            elo2 = sim_elo[team2_idx] + home_advantage
        
        # Calculating Winning Percentage (no ties in the playoffs)
        win_prob1 = 1 / (1 + 10**((elo2 - elo1) / 400))
        
        if random.random() < win_prob1:
            sf_winners.append(team1_idx)
            sf_losers.append(team2_idx)
        else:
            sf_winners.append(team2_idx)
            sf_losers.append(team1_idx)
    
    # Final
    team1_idx = sf_winners[0]
    team2_idx = sf_winners[1]
    
    team1_rank = np.where(sim_playoff_teams == team1_idx)[0][0]
    team2_rank = np.where(sim_playoff_teams == team2_idx)[0][0]
    
    if team1_rank < team2_rank:
        elo1 = sim_elo[team1_idx] + home_advantage
        elo2 = sim_elo[team2_idx]
    else:
        elo1 = sim_elo[team1_idx]
        elo2 = sim_elo[team2_idx] + home_advantage
    
    # Calculating Winning Percentage (no ties in the playoffs)
    win_prob1 = 1 / (1 + 10**((elo2 - elo1) / 400))
    
    if random.random() < win_prob1:
        champion = team1_idx
        runner_up = team2_idx
    else:
        champion = team2_idx
        runner_up = team1_idx
    
    champions[sim] = champion
    finalists[:, sim] = [champion, runner_up]
    semifinalists[:, sim] = sf_winners + sf_losers

champion_probs = np.zeros(n_teams)
semifinal_probs = np.zeros(n_teams)
playoff_probs = np.zeros(n_teams)

for i in range(n_teams):
    champion_probs[i] = np.sum(champions == i) / n_simulations
    semifinal_probs[i] = np.sum(np.any(semifinalists == i, axis=0)) / n_simulations
    playoff_probs[i] = np.sum(np.any(playoff_qualifiers == i, axis=0)) / n_simulations
# 打印所有球队的完整概率
print('\n=== Complete probability analysis for all teams ===')
print('排名\t球队\t预测积分\t季后赛概率\t四强概率\t夺冠概率')
for i, team_idx in enumerate(final_ranking):
    team_name = teams[team_idx]
    final_points = predicted_final_points[team_idx]
    print(f'{i+1}\t{team_name}\t{final_points:.1f}\t\t{playoff_probs[team_idx]*100:.1f}%\t\t{semifinal_probs[team_idx]*100:.1f}%\t\t{champion_probs[team_idx]*100:.1f}%')

print('\n=== Sorted by championship probability ===')
print('球队\t夺冠概率\t四强概率\t季后赛概率')
champion_ranking = np.argsort(champion_probs)[::-1]
for team_idx in champion_ranking:
    if champion_probs[team_idx] > 0 or playoff_probs[team_idx] > 0:  # 只显示有概率的球队
        team_name = teams[team_idx]
        print(f'{team_name}\t{champion_probs[team_idx]*100:.1f}%\t\t{semifinal_probs[team_idx]*100:.1f}%\t\t{playoff_probs[team_idx]*100:.1f}%')

import os
os.makedirs('out', exist_ok=True)

print('\nGenerating charts...')

# 1. All teams probability comparison chart
plt.figure(figsize=(16, 10))
y_pos = np.arange(n_teams)

# Sort by predicted final points
display_ranking = final_ranking
display_teams = [teams[i] for i in display_ranking]
display_playoff = [playoff_probs[i] * 100 for i in display_ranking]
display_semifinal = [semifinal_probs[i] * 100 for i in display_ranking]
display_champion = [champion_probs[i] * 100 for i in display_ranking]

width = 0.25
x = np.arange(len(display_teams))

bars1 = plt.bar(x - width, display_playoff, width, label='季后赛概率', alpha=0.8, color='lightblue', edgecolor='black')
bars2 = plt.bar(x, display_semifinal, width, label='四强概率', alpha=0.8, color='lightcoral', edgecolor='black')
bars3 = plt.bar(x + width, display_champion, width, label='夺冠概率', alpha=0.8, color='gold', edgecolor='black')

for i, (playoff, semifinal, champion) in enumerate(zip(display_playoff, display_semifinal, display_champion)):
    if playoff > 0:
        plt.text(i - width, playoff + 1, f'{playoff:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if semifinal > 0:
        plt.text(i, semifinal + 1, f'{semifinal:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    if champion > 0:
        plt.text(i + width, champion + 1, f'{champion:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xlabel('球队', fontsize=14, fontweight='bold')
plt.ylabel('概率 (%)', fontsize=14, fontweight='bold')
plt.title('江苏省城市足球联赛 - 各队晋级概率完整分析', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, display_teams, rotation=45, fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('out/完整概率分析图.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Complete probability analysis chart saved")

# Champion probability ranking chart
plt.figure(figsize=(14, 10))

champion_data = []
for i in range(n_teams):
    if champion_probs[i] > 0:
        champion_data.append((teams[i], champion_probs[i] * 100))

champion_data.sort(key=lambda x: x[1], reverse=True)

if champion_data:
    champion_teams = [item[0] for item in champion_data]
    champion_values = [item[1] for item in champion_data]
    
    colors = plt.cm.Reds(np.linspace(0.4, 1.0, len(champion_teams)))
    
    y_pos = np.arange(len(champion_teams))
    bars = plt.barh(y_pos, champion_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5, height=0.7)
    
    for i, (bar, prob) in enumerate(zip(bars, champion_values)):
        width = bar.get_width()
        plt.text(width + max(champion_values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}%', ha='left', va='center', fontsize=12, fontweight='bold')
    
    plt.yticks(y_pos, champion_teams, fontsize=12)
    plt.xlabel('夺冠概率 (%)', fontsize=14, fontweight='bold')
    plt.ylabel('球队', fontsize=14, fontweight='bold')
    plt.title('江苏省城市足球联赛 - 各队夺冠概率排行榜', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.xlim(0, max(champion_values) * 1.15)
    
    plt.gca().invert_yaxis()
    
else:
    plt.text(0.5, 0.5, '暂无球队具有夺冠概率\n请完成更多模拟', 
             ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
    plt.title('江苏省城市足球联赛 - 各队夺冠概率排行榜', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('out/夺冠概率排行榜.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Championship probability ranking chart saved")

# Points comparison chart
plt.figure(figsize=(14, 10))
y_pos = np.arange(n_teams)

width = 0.4
y1 = y_pos - width/2 
y2 = y_pos + width/2  

bars1 = plt.barh(y1, [current_points[i] for i in final_ranking], height=width, 
                label='当前积分', color='steelblue', alpha=0.8, edgecolor='black')
bars2 = plt.barh(y2, [predicted_final_points[i] for i in final_ranking], height=width,
                label='预测最终积分', color='orange', alpha=0.8, edgecolor='black')

for i, team_idx in enumerate(final_ranking):
    current = current_points[team_idx]
    predicted = predicted_final_points[team_idx]
    
    plt.text(current + 0.5, y1[i], f'{current:.0f}', va='center', ha='left', 
             fontsize=11, fontweight='bold', color='darkblue')
    
    plt.text(predicted + 0.5, y2[i], f'{predicted:.1f}', va='center', ha='left', 
             fontsize=11, fontweight='bold', color='darkorange')

plt.xlabel('积分', fontsize=14, fontweight='bold')
plt.ylabel('球队', fontsize=14, fontweight='bold')
plt.title('江苏省城市足球联赛 - 常规赛积分预测对比', fontsize=16, fontweight='bold', pad=20)
plt.yticks(y_pos, [teams[i] for i in final_ranking], fontsize=12)
plt.legend(fontsize=12, loc='lower right')
plt.grid(True, alpha=0.3, axis='x')
plt.xlim(0, max(predicted_final_points) + 5)
plt.tight_layout()
plt.savefig('out/积分对比图.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Points comparison chart saved")

# Elo rating chart
plt.figure(figsize=(14, 8))
elo_ranking = np.argsort(current_elo)[::-1]
colors = plt.cm.viridis(np.linspace(0, 1, n_teams))

bars = plt.bar(range(n_teams), [current_elo[i] for i in elo_ranking], 
               color=colors, alpha=0.8, edgecolor='black', linewidth=1)

for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('球队', fontsize=14, fontweight='bold')
plt.ylabel('Elo评分', fontsize=14, fontweight='bold')
plt.title('江苏省城市足球联赛 - 各队当前Elo评分', fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(n_teams), [teams[i] for i in elo_ranking], rotation=45, ha='right', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('out/Elo评分图.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Elo rating chart saved")

# Playoff probability heatmap
if any(playoff_probs > 0):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    prob_data = []
    team_labels = []
    for i, team_idx in enumerate(final_ranking):
        if playoff_probs[team_idx] > 0 or i < 10:  
            prob_data.append([
                playoff_probs[team_idx] * 100,
                semifinal_probs[team_idx] * 100,
                champion_probs[team_idx] * 100
            ])
            team_labels.append(teams[team_idx])
    
    prob_array = np.array(prob_data)
    
    im = ax.imshow(prob_array, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(3))
    ax.set_xticklabels(['季后赛概率', '四强概率', '夺冠概率'], fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(team_labels)))
    ax.set_yticklabels(team_labels, fontsize=12)
    
    for i in range(len(team_labels)):
        for j in range(3):
            if prob_array[i, j] > 0:
                text = ax.text(j, i, f'{prob_array[i, j]:.1f}%',
                             ha="center", va="center", color="black" if prob_array[i, j] < 50 else "white",
                             fontweight='bold', fontsize=10)
    
    ax.set_title("江苏省城市足球联赛 - 各队晋级概率热力图", fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im)
    cbar.set_label('概率 (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('out/概率热力图.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Probability heatmap saved")

# Generate prediction result CSV file
print('\nGenerating prediction results CSV file...')

csv_data = []
for i, team_idx in enumerate(final_ranking):
    team_name = teams[team_idx]
    current_pts = current_points[team_idx]
    predicted_final_pts = predicted_final_points[team_idx]
    current_games = games_played[team_idx]
    remaining_games_count = 12 - current_games
    team_wins = wins[team_idx]
    team_draws = draws[team_idx]
    team_losses = losses[team_idx]
    team_elo = current_elo[team_idx]
    playoff_prob = playoff_probs[team_idx] * 100
    semifinal_prob = semifinal_probs[team_idx] * 100
    champion_prob = champion_probs[team_idx] * 100
    
    csv_data.append({
        '排名': i + 1,
        '球队': team_name,
        '当前积分': int(current_pts),
        '已赛场次': int(current_games),
        '剩余场次': int(remaining_games_count),
        '胜': int(team_wins),
        '平': int(team_draws),
        '负': int(team_losses),
        '当前Elo评分': round(team_elo, 1),
        '预测最终积分': round(predicted_final_pts, 1),
        '季后赛概率(%)': round(playoff_prob, 1),
        '四强概率(%)': round(semifinal_prob, 1),
        '夺冠概率(%)': round(champion_prob, 1)
    })

import pandas as pd
df_predictions = pd.DataFrame(csv_data)

csv_filename = 'out/江苏省城市足球联赛预测结果.csv'
df_predictions.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"Prediction results CSV saved: {csv_filename}")
print('\n' + '='*80)
print('江苏省城市足球联赛预测结果汇总表')
print('='*80)
print(df_predictions.to_string(index=False))
print('='*80)

# 显示所有图表
plt.show()

print('\n All analysis completed')
print(' Prediction results saved to out/ directory')
print('\n Program execution finished')
