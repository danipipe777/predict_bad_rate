import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(
        matrix:list,
        model:str,
        ths:np.ndarray,
        index_best_th=300,
        start=0,
        end=500
):
    '''
    Useful Function to plot a confusion matrix with a threshold slider

    Params:
        matrix: list of nxn confusion matrices,
        model: string with model name,
        ths: np.ndarray of thresholds normally np.linspace(0, 1, 1000),
        index_best_th: index that yields best ths (using best f1)
    '''

    fig = go.Figure()

    for i in range(start, end):
        table_data = np.vstack([
            np.array([['label_0', 'label_1']]),
            matrix[i].astype(str).T
        ])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['', 'Predict 0', 'Predict 1'],
                    font=dict(size=16),
                    fill_color='DarkGray'
                ),
                cells=dict(
                    values=table_data,
                    font=dict(size=15),
                    align=['left', 'center', 'center'],
                    height=30,
                    fill_color=['DarkGray', 'AliceBlue']
                ),
                visible=False
            )
        )

    ## Create sliders
    steps = []
    for i, val in enumerate(ths[start:end]):
        current_matrix = matrix[i]
        TP, FP, TN, FN = (
            current_matrix[1, 1],
            current_matrix[0, 1],
            current_matrix[0, 0],
            current_matrix[1, 0],
        )

        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        bad_rate = FN/(TN + FN)

        step = dict(
            method="update",
            args=[{"visible": [False] * len(ths[start:end])},
                  {"title": (
                      model + " threshold: " + '{:.4f}'.format(val) + ", " +
                       f'precision: {precision:.2f}' + " " +
                       f'recall: {recall:.2f}' + " " +
                       f'bad_rate: {bad_rate:.2f}'
                  ), 'font':{'size':12} }],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=max(index_best_th-start, 0),
        steps=steps
    )]


    fig.data[index_best_th-start].visible = True

    fig.update_layout(
        sliders=sliders,
        margin=dict(t=50, l=50, r=50, b=100),
        width=800, height=250,
        title= model + " threshold: " +'{:.4f}'.format(ths[index_best_th])
    )

    return fig

def add_height_text(ax, fontsize=16):
    for patch in ax.patches:
        height, x, width = patch.get_height(), patch.get_x(), patch.get_width()
        y_lims = ax.get_ylim()
        ax_height = y_lims[1] - y_lims[0]

        x_lims = ax.get_xlim()
        ax_width = x_lims[1] - x_lims[0]

        start = x
        end = x + width
        middle = (start+end)/2

        text = str(height)
        text_len = len(text)

        rate = 1/(16 * 190)

        ax.text(
            middle - text_len*ax_width*fontsize*rate,
            height + ax_height*0.01,
            text,
            fontsize=fontsize
        )
