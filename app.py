import streamlit as st
import numpy as np
import random

# Game config
GRID_SIZE = 8
goal = (GRID_SIZE - 1, GRID_SIZE - 1)
actions = ['up', 'down', 'left', 'right']
action_idx = {a: i for i, a in enumerate(actions)}
obstacles = [(2, 2), (3, 3), (1, 5), (5, 1), (6, 6)]

# Session state initialization
if 'player_pos' not in st.session_state:
    st.session_state.player_pos = (0, 0)
    st.session_state.agent_pos = (0, GRID_SIZE - 1)
    st.session_state.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(actions)))
    st.session_state.epsilon = 0.2
    st.session_state.game_over = False
    st.session_state.player_score = 0
    st.session_state.ai_score = 0

# Movement logic
def step(pos, action):
    i, j = pos
    if action == 'up': i = max(i - 1, 0)
    elif action == 'down': i = min(i + 1, GRID_SIZE - 1)
    elif action == 'left': j = max(j - 1, 0)
    elif action == 'right': j = min(j + 1, GRID_SIZE - 1)
    new_pos = (i, j)
    return new_pos if new_pos not in obstacles else pos

def choose_action(state):
    if np.random.rand() < st.session_state.epsilon:
        return random.choice(actions)
    return actions[np.argmax(st.session_state.q_table[state[0], state[1]])]

def update_q(agent_pos, action, reward, new_pos):
    q = st.session_state.q_table
    a_idx = action_idx[action]
    old = q[agent_pos[0], agent_pos[1], a_idx]
    future = np.max(q[new_pos[0], new_pos[1]])
    q[agent_pos[0], agent_pos[1], a_idx] = old + 0.1 * (reward + 0.9 * future - old)

def reset_positions():
    st.session_state.player_pos = (0, 0)
    st.session_state.agent_pos = (0, GRID_SIZE - 1)
    st.session_state.game_over = False

def full_restart():
    st.session_state.player_score = 0
    st.session_state.ai_score = 0
    reset_positions()

def draw_grid_table():
    grid = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            pos = (i, j)
            if pos == st.session_state.player_pos:
                row.append('🧍')
            elif pos == st.session_state.agent_pos:
                row.append('🤖')
            elif pos == goal:
                row.append('🎯')
            elif pos in obstacles:
                row.append('🟥')
            else:
                row.append('⬜')
        grid.append(row)
    return grid

st.set_page_config(page_title="GridWorld RL Game", layout="centered")
st.title("🧠 GridWorld RL: Player vs AI")
st.subheader("🎯 Reach the Goal • Avoid the Obstacles")

st.markdown(f"**🧍 Player Score:** {st.session_state.player_score} | **🤖 AI Score:** {st.session_state.ai_score}")

grid_table = draw_grid_table()
st.table(grid_table)

if not st.session_state.game_over:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("Move:")
        if st.button("⬆️", key="up"):
            st.session_state.player_pos = step(st.session_state.player_pos, 'up')
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("⬅️", key="left"):
                st.session_state.player_pos = step(st.session_state.player_pos, 'left')
        with c2:
            if st.button("⬇️", key="down"):
                st.session_state.player_pos = step(st.session_state.player_pos, 'down')
        with c3:
            if st.button("➡️", key="right"):
                st.session_state.player_pos = step(st.session_state.player_pos, 'right')

    # Check if player wins
    if st.session_state.player_pos == goal:
        st.session_state.player_score += 1
        st.success("🧍 Player wins!")
        reward = -1
        st.session_state.game_over = True
    else:
        # AI move
        action = choose_action(st.session_state.agent_pos)
        new_agent_pos = step(st.session_state.agent_pos, action)
        if new_agent_pos == goal:
            st.session_state.ai_score += 1
            st.success("🤖 AI wins!")
            reward = 1
            st.session_state.game_over = True
        else:
            reward = -0.01
        update_q(st.session_state.agent_pos, action, reward, new_agent_pos)
        st.session_state.agent_pos = new_agent_pos
        st.session_state.epsilon = max(0.05, st.session_state.epsilon * 0.999)

# Restart or End Game
b1, b2 = st.columns(2)
with b1:
    if st.button("🔄 Restart"):
        reset_positions()
        st.rerun()
with b2:
    if st.button("🏁 End Game"):
        if st.session_state.player_score > st.session_state.ai_score:
            st.balloons()
            st.success("🧍 Player wins the session!")
        elif st.session_state.ai_score > st.session_state.player_score:
            st.error("🤖 AI wins the session!")
        else:
            st.info("⚖️ It's a tie!")
        full_restart()
        st.rerun()

st.markdown("**Use the buttons above to control the player.**")
