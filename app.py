import streamlit as st
import pandas as pd
import numpy as np
from algorithm import DecisionTree
from visualizer import plot_decision_tree
from utils import generate_sample_data

st.set_page_config(page_title="Decision Tree Visualizer", layout="wide")

def main():
    st.title("Decision Tree Visualizer")
    st.write("Learn how a decision tree works with this interactive tool!")

    # Sidebar for configuration
    st.sidebar.header("Settings")
    max_depth = st.sidebar.slider("Tree Depth", 1, 5, 2)
    min_samples_split = st.sidebar.slider("Min Samples to Split", 2, 10, 2)

    # Use sample dataset
    data = generate_sample_data()
    st.write("Sample Dataset (Iris-like):")
    st.dataframe(data.head())

    # Prepare data
    X = data[['feature1', 'feature2']].values
    y = data['target'].values

    # Train decision tree
    try:
        dt = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
        dt.fit(X, y)

        # Create tabs for views
        tab1, tab2, tab3 = st.tabs(["Tree Visualization", "Steps", "Analysis"])

        with tab1:
            st.subheader("Decision Tree")
            fig = plot_decision_tree(dt, ['Feature1', 'Feature2'], ['Class 0', 'Class 1'])
            st.pyplot(fig)

        with tab2:
            st.subheader("How It Works")
            steps = dt.get_steps()
            for i, step in enumerate(steps):
                with st.expander(f"Step {i+1}: {step['action']}"):
                    st.write(f"Node: {step['node']}")
                    st.write(f"Gini: {step['gini']:.3f}")
                    if 'split' in step:
                        st.write(f"Split on: Feature {step['split']['feature']} <= {step['split']['threshold']:.2f}")

        with tab3:
            st.subheader("Complexity Analysis")
            st.markdown("""
            **Time Complexity:**
            - Training: O(n * m * log(n)) (n = samples, m = features)
            - Prediction: O(log(n))

            **Space Complexity:**
            - O(n) for tree structure

            **Test Cases:**
            - Best: Clear class separation
            - Average: Mixed classes
            - Worst: Overlapping classes
            """)

    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()