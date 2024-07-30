from flask import Flask, render_template, request, send_from_directory
import pickle
from pgmpy.models import BayesianNetwork
import pandas as pd
import networkx as nx
from pgmpy.inference import VariableElimination
from matplotlib import pyplot as plt
import os
import numpy as np

app = Flask(__name__)

# Ensure directory for images exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load the saved model
with open('bayesian_network_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize inference
infer = VariableElimination(model)


@app.route('/', methods=['GET', 'POST'])
def index():
    query_variables = list(model.nodes())  # Assuming variables are the node names of the Bayesian Network

    # Get evidence options
    evidence_options = {}
    for node in model.nodes():
        cpds = model.get_cpds(node)
        state_names = cpds.state_names[node]
        evidence_options[node] = state_names

    if request.method == 'POST':
        query_variable = request.form.get('query_variable')
        evidence_vars = request.form.getlist('evidence_var[]')  # Retrieve list of evidence inputs

        print(f"Query Variable: {query_variable}")
        print(f"Evidences: {evidence_vars}")

        # Parse evidence into a dictionary
        evidence_dict = {}
        for item in evidence_vars:
            if '=' in item:
                key, value = item.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key != query_variable:
                    evidence_dict[key] = value

        # Validate form data
        if query_variable and evidence_dict is not None:
            # Query the model
            try:
                # Query the model
                query_result = infer.query(variables=[query_variable], evidence=evidence_dict)

                # Save plots
                histogram_img, piechart_img, query_title, evidence_title = save_plots(query_result, query_variable, evidence_dict)

                # Pass the images and query variables to the template
                return render_template('index.html',
                                       histogram_img=histogram_img,
                                       piechart_img=piechart_img,
                                       query_variables=query_variables,
                                       evidence_options=evidence_options,
                                       query_title=query_title,
                                       evidence_title=evidence_title)
            except ValueError as e:
                # Handle query errors
                return f"Error: {e}"

    return render_template('index.html', query_variables=query_variables, evidence_options=evidence_options)


@app.route('/static/<filename>')
def serve_image(filename):
    return send_from_directory('static', filename)


def save_plots(query_result, query_variable, evidence_dict):
    probabilities = query_result.values
    categories = query_result.state_names[query_result.variables[0]]

    # Filter out zero probabilities for better readability
    nonzero_indices = np.where(probabilities > 0)
    filtered_probabilities = probabilities[nonzero_indices]
    filtered_categories = np.array(categories)[nonzero_indices]

    # Titles based on query and evidence
    query_title = f"{query_variable}"
    evidence_title = ', '.join([f"{k}={v}" for k, v in evidence_dict.items()])

    # Plot and save Pie Chart
    plt.figure(figsize=(16, 8))
    plt.pie(filtered_probabilities, labels=filtered_categories, autopct='%1.1f%%', radius=0.8, startangle=140,
            colors=plt.cm.Paired(range(len(filtered_categories))))
    # plt.title(f"Probability Distribution for '{query_variable}' variable, given '{evidence_title}'")
    plt.legend(filtered_categories, title='Performance Categories', loc='best', bbox_to_anchor=(1, 0.5))

    # Adjust margins to make space for the legend
    plt.subplots_adjust(right=1)
    plt.tight_layout(pad=2.0)

    # Save Pie Chart
    piechart_path = 'static/piechart.png'
    plt.savefig(piechart_path)
    plt.clf()  # Clear the current figure
    # Plot and save Histogram
    plt.figure(figsize=(16, 8))
    plt.bar(categories, probabilities, color='lightgreen')
    plt.xlabel('Performance Category')
    plt.ylabel('Probability')
    # plt.title(f"Histogram for probability distribution of '{query_title}' variable, given '{evidence_title}'")
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
    plt.tight_layout(pad=2.0)  # Increase padding to avoid clipping

    # Save Histogram
    histogram_path = 'static/histogram.png'
    plt.savefig(histogram_path)
    plt.clf()  # Clear the current figure
    return 'histogram.png', 'piechart.png', query_title, evidence_title


@app.route('/structure')
def structure():
    # Convert BayesianNetwork to NetworkX graph
    # with open('bayesian_network_model.pkl', 'rb') as f:
    #     model = pickle.load(f)

    # Initialize inference
    graph = nx.DiGraph(model.edges())
    pos = nx.spring_layout(graph)  # You can adjust the seed for different layouts

    # Draw the graph with improved settings
    plt.figure(figsize=(16, 8))  # Increase figure size if needed
    nx.draw(graph, pos, with_labels=True, node_size=4000, node_color='skyblue', font_size=15, font_weight='bold', edge_color='black', arrowsize=20)
    plt.title('Bayesian Network Structure')
    network_img_path = 'static/network.png'
    plt.tight_layout(pad=2.0)
    plt.savefig(network_img_path)
    plt.clf()  # Clear the current figure
    return render_template('structure.html', network_img='network.png')
    # plt.show()


@app.route('/cpdVis')
def cpdVis():
    cpds = model.get_cpds()
    image_files = []
    for i, cpd in enumerate(cpds):
        print(f'Length of cpd{i} is {len(cpd.variables)}')
        if len(cpd.variables) > 3:
            continue
        image_path = f'./static/cpd_{i}.png'
        plot_cpd(cpd, image_path)
        image_files.append(f'cpd_{i}.png')
    print(image_files)
    return render_template('cpds.html', image_files=image_files)


def plot_cpd(cpd, image_path):
    """
    Plot the CPD of a given TabularCPD object and save it to an image file.
    """
    variable = cpd.variable
    state_names = cpd.state_names
    values = cpd.values.flatten()

    if len(cpd.variables) == 1:
        # For single-variable CPD
        df = pd.DataFrame({'State': state_names[variable], 'Probability': values})
        plt.figure(figsize=(12, 8))
        plt.bar(df['State'], df['Probability'], color='green')
        plt.xlabel(variable)
        plt.ylabel('Probability')
        plt.title(f'CPD of {variable}')
        plt.xticks(rotation=45)
        plt.tight_layout(pad=2.0)
        plt.savefig(image_path, bbox_inches='tight')
        plt.clf()

    elif len(cpd.variables) == 2:
        # For two-variable CPD (conditional)
        var1, var2 = cpd.variables
        state_names_var1 = state_names[var1]
        state_names_var2 = state_names[var2]

        df = pd.DataFrame(values.reshape(len(state_names_var1), len(state_names_var2)), index=state_names_var1,
                          columns=state_names_var2)

        plt.figure(figsize=(12, 8))
        cax = plt.matshow(df, cmap='Blues')
        plt.colorbar(cax)
        plt.title(f'CPD of {var1} given {var2}')
        plt.xlabel(var2)
        plt.ylabel(var1)
        plt.xticks(ticks=np.arange(len(state_names_var2)), labels=state_names_var2, rotation=45)
        plt.yticks(ticks=np.arange(len(state_names_var1)), labels=state_names_var1)
        plt.tight_layout(pad=2.0)
        plt.savefig(image_path, bbox_inches='tight')
        plt.clf()

    elif len(cpd.variables) == 3:
        # For three-variable CPD
        var1, var2, var3 = cpd.variables
        state_names_var1 = state_names[var1]
        state_names_var2 = state_names[var2]
        state_names_var3 = state_names[var3]

        # Reshape values for plotting
        values_reshaped = values.reshape(len(state_names_var1), len(state_names_var2), len(state_names_var3))

        # Plot each combination of the first two variables
        for i, state1 in enumerate(state_names_var1):
            plt.figure(figsize=(12, 8))
            plt.bar(state_names_var3, values_reshaped[i].sum(axis=0), color='green')
            plt.xlabel(var3)
            plt.ylabel('Probability')
            plt.title(f'CPD of {var2} given {var1} = {state1}')
            plt.xticks(rotation=45)
            plt.tight_layout(pad=2.0)
            plt.savefig(f"{image_path}_{i}.png", bbox_inches='tight')
            plt.clf()


if __name__ == '__main__':
    app.run(debug=True)
