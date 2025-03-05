import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import shap

def plot_shap_beeswarm(shap_values):
    # Define the color palette
    start_color = (0.34, 0.79, 0.55)  # Green
    middle_color = (0.98, 0.82, 0.22)  # Yellow
    end_color = (0.95, 0.60, 0.19)  # Orange
    colors = [start_color, middle_color, end_color]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, 256)

    plt.figure(figsize=(5, 6))  # Adjust height as needed.

    # Plot the beeswarm
    ax = shap.plots.beeswarm(
        shap_values,
        show=False,
        max_display=25,
        color=cmap,
        color_bar=False,
        plot_size=(5, 25),
    )

    # Get the current y-axis tick labels (feature names)
    y_ticks_labels = [label.get_text() for label in ax.get_yticklabels()]

    # Modify the feature names
    modified_labels = [label.replace("_", " ").title() for label in y_ticks_labels]

    # Set the modified labels back to the y-axis
    ax.set_yticklabels(modified_labels)

    # Add horizontal lines
    # Get the y-axis tick positions
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        # Add a dashed gray line
        ax.axhline(y=y, color="gray", linestyle="-", linewidth=0.5)

    # Add colorbar above chart
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
        ),
        ax=ax,
        location="top",
        shrink=0.5,
        aspect=20,
        ticks=[],
        pad=0.02
    )
    cbar.outline.set_visible(False)
    cbar.set_label("Feature Value", loc="center", labelpad=10)
    # Add labels to the left and right of the colorbar
    cbar.ax.text(
        -0.1,  # Adjust horizontal position for left label
        0.5,  # Vertical position (middle of colorbar)
        "Low",  # Left label text
        va="center",
        ha="right",
        transform=cbar.ax.transAxes,
    )
    cbar.ax.text(
        1.1,  # Adjust horizontal position for right label
        0.5,  # Vertical position (middle of colorbar)
        "High",  # Right label text
        va="center",
        ha="left",
        transform=cbar.ax.transAxes,
    )
    return plt.gcf()