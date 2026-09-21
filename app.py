import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Smart Warehouse Robot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Smart Warehouse Robot: Q-Learning Demo")
st.write("This app loads your trained Q-learning model and simulates the robot navigating through the warehouse grid to reach the charging station (**GOAL**).")

# Load the trained model from the pickle file
@st.cache_resource
def load_model():
    try:
        with open('warehouse_robot_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        return None

model_data = load_model()

if model_data is None:
    st.error("⚠️ Model file `warehouse_robot_model.pkl` not found! Please make sure you train and save the model first.")
else:
    Q = model_data['Q_table']
    action_to_id = model_data['action_to_id']
    id_to_action = model_data['id_to_action']

    st.success("✅ Model loaded successfully from `warehouse_robot_model.pkl`!")

    # Display Warehouse Grid Layout
    st.subheader("Warehouse Grid Layout (4x4)")
    grid_display = np.array([
        ["S0", "S1", "S2", "S3"],
        ["S4", "S5", "S6", "S7"],
        ["S8", "S9", "S10", "S11"],
        ["S12", "S13", "S14", "GOAL (S15)"]
    ])
    st.table(grid_display)

    # Helper function to compute next state based on grid movement rules
    def get_next_state(state, action_name):
        row, col = state // 4, state % 4
        if action_name == 'LEFT' and col > 0:
            return state - 1
        elif action_name == 'RIGHT' and col < 3:
            return state + 1
        elif action_name == 'UP' and row > 0:
            return state - 4
        elif action_name == 'DOWN' and row < 3:
            return state + 4
        return state # Boundary hit, stays in the same state

    # Simulation Controls
    st.subheader("Run Simulation")
    start_state = st.selectbox("Select Robot Start State:", list(range(15)), index=0)

    if st.button("Start Robot Navigation"):
        current_state = start_state
        path = [current_state]
        actions_taken = []
        steps = 0
        
        # Run simulation loop
        while current_state != 15 and steps < 20:
            best_action_id = np.argmax(Q[current_state])
            best_action_name = id_to_action[best_action_id]
            
            actions_taken.append(best_action_name)
            current_state = get_next_state(current_state, best_action_name)
            path.append(current_state)
            steps += 1

        # Display Results
        st.write("---")
        st.write(f"**Starting State:** S{start_state}")
        st.write(f"**Path Taken (States):** { [f'S{s}' for s in path] }")
        st.write(f"**Actions Executed:** {actions_taken}")
        
        if current_state == 15:
            st.success("🎉 Success! The robot successfully reached the Charging Station (GOAL / S15).")
        else:
            st.warning("⚠️ The robot reached the step limit without hitting the goal.")

    # Show Raw Q-Table
    with st.expander("View Raw Q-Table"):
        st.write(Q)
