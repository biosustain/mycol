"""Dependency-light Plotly helpers shared across the app's charts.

Kept as a leaf module (plotly only, no other app imports) so any module can use
it without creating an import cycle — e.g. ``mask_editing_functions`` already
imports ``sam2_functions``. Also the single source of truth for the plot palette.
"""

import plotly.graph_objects as go

# --- Plot colour palette (single source of truth; reused across every chart) ---
NAVY = "#004280"        # primary dark blue: markers, lines, bar/error-bar outlines
PALE_BLUE = "#EBF1F8"   # bar fill
LIGHT_BLUE = "#D3E4F4"  # train-loss line and markers

MIN_DRAG_PX = 3  # smallest drag, in display pixels, that counts as a real box


def make_base_figure(bg_img, disp_w: int, disp_h: int, dragmode: str) -> go.Figure:
    """
    Create a Plotly figure with a background image and fixed pixel size.
    Used by the box, freehand and ellipse drawing modes.
    """
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=bg_img,
            xref="x",
            yref="y",
            x=0,
            y=disp_h,
            sizex=disp_w,
            sizey=disp_h,
            sizing="stretch",
            layer="below",
        )
    )
    fig.update_xaxes(visible=False, range=[0, disp_w], constrain="domain")
    fig.update_yaxes(
        visible=False,
        range=[0, disp_h],
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        dragmode=dragmode,
        # "d" keeps thin drags as boxes; "any" snaps them to a full row or column
        selectdirection="d",
        margin=dict(l=0, r=0, t=0, b=0),
        width=disp_w,
        height=disp_h,
    )
    return fig


def point_hover_texts(numbers, names, patches=None):
    """Per-point hover labels — image number, image name and (optionally) patch
    number, each on its own line. Shared by the cell-metrics and fine-tuning plots."""
    if patches is None:
        patches = [None] * len(names)
    texts = []
    for num, name, patch in zip(numbers, names, patches):
        line = f"Image number: {num}<br>Image name: {name}"
        if patch is not None:
            line += f"<br>Patch number: {patch}"
        texts.append(line)
    return texts


def plot_loss_curve(train_losses, test_losses):
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_scatter(
        x=epochs,
        y=train_losses,
        mode="lines+markers",
        name="train",
        line=dict(color=LIGHT_BLUE, width=2),
        marker=dict(color=LIGHT_BLUE, size=6),
    )

    # Cellpose only evaluates validation loss every 10 epochs; skip zero entries
    val_pairs = [(i + 1, v) for i, v in enumerate(test_losses) if v != 0]
    e_val = [p[0] for p in val_pairs]
    val_scores = [p[1] for p in val_pairs]
    fig.add_scatter(
        x=e_val,
        y=val_scores,
        mode="lines+markers",
        name="val",
        line=dict(color=NAVY, width=2),
        marker=dict(color=NAVY, size=6),
    )
    fig.update_layout(
        title="Training vs. Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        width=450,
    )
    return fig


def plot_confusion_matrix(cm, class_names):
    n = len(class_names)
    text = [[f"{cm[i,j]}" for j in range(n)] for i in range(n)]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=text,
            textfont=dict(size=20),
            texttemplate="%{text}",
            colorscale="Blues",
            hoverongaps=False,
            showscale=False,
        )
    )
    fig.update_layout(
        title="Class Confusion Matrix",
        xaxis=dict(title="Predicted Class", tickangle=45),
        yaxis=dict(title="True Class", autorange="reversed"),
        width=max(500, 80 * n),
        height=max(400, 80 * n),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=40, b=80),
    )
    return fig


def plot_densenet_metrics(metrics):
    labels, values = list(metrics.keys()), list(metrics.values())
    fig = go.Figure(layout=dict(barcornerradius=10))
    fig.add_bar(
        x=labels,
        y=values,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        marker=dict(color=[PALE_BLUE] * 4, line=dict(color=NAVY, width=2)),
        name="metrics",
    )
    fig.update_yaxes(range=[0, 1.2])
    fig.update_layout(
        title="Validation metrics",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=450,
        width=450,
    )
    return fig
