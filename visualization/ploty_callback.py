from IPython.display import display, clear_output, HTML
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pytorch_lightning as pl
import plotly.io as pio
from PIL import Image
import io
from typing import Dict, Optional
from matplotlib.colors import ListedColormap
# Updated imports: Added field
from dataclasses import dataclass, field
from typing import Tuple, Union, List
from collections import defaultdict
import ipywidgets as widgets


# Custom colormap
high_contrast_colors = [
    "#7F3C8D", "#1CA878", "#3969AC", "#F2B701", "#D63F6C",
    "#A0C95A", "#E68310", "#008695", "#CF1C90", "#005082"
]
# It's good practice to define the default colormap once
# if it's created the same way every time. The factory will use this list.
default_cmap_colors = ListedColormap(high_contrast_colors)

@dataclass
class PlotConfig:
    show_plots: bool = False
    selected_plots: List[str] = None
    figure_size: Tuple[int, int] = (5, 5)
    dpi: int = 150
    # Corrected line: Use default_factory for the mutable ListedColormap
    cmap: Union[str, ListedColormap] = field(default_factory=lambda: ListedColormap(high_contrast_colors))

    def __post_init__(self):
        if self.selected_plots is None:
            self.selected_plots = ['embeddings', 'cluster_sizes']

# --- The rest of your code (BasePlot, EmbeddingsPlot, etc.) remains the same ---

class BasePlot:
    def __init__(self, config: PlotConfig):
        self.config = config

    def plot(self, data: Dict) -> go.Figure:
        raise NotImplementedError

class EmbeddingsPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        embeddings = np.array(data['embeddings'])
        labels = np.array(data['labels'])

        if embeddings.shape[1] == 2:
            return self.plot_2d(embeddings, labels)
        elif embeddings.shape[1] == 3:
            return self.plot_3d(embeddings, labels)
        else:
            raise ValueError("Embedding dimension must be 2 or 3.")

    def plot_2d(self, embeddings: np.ndarray, labels: np.ndarray) -> go.Figure:
        labels_str = labels.astype(str)
        # Sort data by label to ensure consistent color assignment
        sorted_indices = np.argsort(labels_str)

        # Use the cmap from the config if it's a ListedColormap, otherwise use the high_contrast_colors list directly
        # Plotly express handles ListedColormap or list of colors in color_discrete_sequence
        color_sequence = self.config.cmap.colors if isinstance(self.config.cmap, ListedColormap) else self.config.cmap
        if not isinstance(color_sequence, list): # If it was a string name like 'viridis'
             color_sequence = high_contrast_colors # Fallback or handle string names appropriately

        fig = px.scatter(
            x=embeddings[sorted_indices, 0],
            y=embeddings[sorted_indices, 1],
            color=labels_str[sorted_indices],
            color_discrete_sequence=color_sequence,
        )

        fig.update_layout(
            title="2D Embeddings Visualization",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            template="simple_white",
            legend_title="Label"
        )
        fig.update_traces(marker=dict(size=4, opacity=0.5))
        return fig

    def plot_3d(self, embeddings: np.ndarray, labels: np.ndarray) -> go.Figure:
        fig = go.Figure()
        labels_str = labels.astype(str)
        unique_labels = sorted(np.unique(labels_str), key=lambda x: int(x) if x.isdigit() else x)

        # Get color list
        color_sequence = self.config.cmap.colors if isinstance(self.config.cmap, ListedColormap) else self.config.cmap
        if not isinstance(color_sequence, list): # If it was a string name like 'viridis'
             color_sequence = high_contrast_colors # Fallback or handle string names appropriately

        for i, label in enumerate(unique_labels):
            mask = labels_str == label
            fig.add_trace(go.Scatter3d(
                x=embeddings[mask, 0],
                y=embeddings[mask, 1],
                z=embeddings[mask, 2],
                mode='markers',
                name=label,
                marker=dict(
                    size=4,
                    opacity=0.8,
                    # Use the color sequence from config
                    color=color_sequence[i % len(color_sequence)]
                )
            ))

        # Optional: grid lines (sphere)
        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 18)
        for phi in v:
            x = np.cos(u) * np.sin(phi)
            y = np.sin(u) * np.sin(phi)
            z = np.full_like(u, np.cos(phi))
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='lightgray', width=1), showlegend=False)) # Lighter gray
        for theta in u:
            x = np.cos(theta) * np.sin(v)
            y = np.sin(theta) * np.sin(v)
            z = np.cos(v)
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='lightgray', width=1), showlegend=False)) # Lighter gray


        fig.update_layout(
            title="3D Embeddings on Unit Sphere",
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)"),
                aspectmode='data', # Ensures sphere looks like a sphere
                bgcolor="rgba(0,0,0,0)" # Transparent background for the scene
            ),
            template="simple_white",
            showlegend=True,
            legend_title="Label",
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background for the whole plot
            plot_bgcolor='rgba(0,0,0,0)'
        )
        # Try to make the sphere look better by setting camera position
        fig.update_layout(scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.5) # Adjust camera angle
        ))
        return fig


class ClusterSizesPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        probabilities = torch.tensor(data['probabilities'])
        # Ensure probabilities is at least 2D if it's 1D
        if probabilities.ndim == 1:
             probabilities = probabilities.unsqueeze(0)
        if probabilities.numel() == 0: # Handle empty tensor case
            fig = go.Figure()
            fig.update_layout(title='Cluster Sizes with Uncertainty (No Data)')
            return fig

        cluster_sizes = probabilities.sum(dim=0)
        # Handle case where sum results in 0 dimension tensor (only one data point)
        if cluster_sizes.ndim == 0:
            cluster_sizes = cluster_sizes.unsqueeze(0)
            # Uncertainty calculation might need adjustment or skipping if only 1 data point
            if probabilities.size(0) > 0 :
                 uncertainty = torch.sqrt((probabilities * (1 - probabilities)).sum(dim=0) / probabilities.size(0))
                 if uncertainty.ndim == 0: uncertainty = uncertainty.unsqueeze(0)
            else:
                 uncertainty = torch.zeros_like(cluster_sizes)
        else:
             uncertainty = torch.sqrt((probabilities * (1 - probabilities)).sum(dim=0) / probabilities.size(0))


        cluster_sizes, indices = torch.sort(cluster_sizes, descending=True)
        uncertainty = uncertainty[indices]

        fig = go.Figure(data=[
            go.Bar(
                x=[f"Cluster {i}" for i in indices.numpy()], # Use original indices for labels if needed, or just sorted index
                # x=np.arange(len(cluster_sizes)), # Or keep numeric index
                y=cluster_sizes.numpy(),
                error_y=dict(type='data', array=uncertainty.numpy(), visible=True),
                marker_color='skyblue'
            )
        ])

        fig.update_layout(
            title='Cluster Sizes with Uncertainty',
            xaxis_title='Cluster Index (Sorted by Size)',
            yaxis_title='Number of Points',
            template='simple_white'
        )
        return fig

class NeighborhoodDistPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        learned_tensor = data.get('learned_distribution')
        supervisory_tensor = data.get('supervisory_distribution')

        if learned_tensor is None or supervisory_tensor is None:
             fig = go.Figure()
             fig.update_layout(title='Neighbor Selection Probability (Data Missing)')
             return fig

        learned = torch.nn.functional.normalize(learned_tensor, p=1, dim=-1)
        supervisory = torch.nn.functional.normalize(supervisory_tensor, p=1, dim=-1)

        # Handle potential NaN/Inf after normalization if input rows were all zero
        learned = torch.nan_to_num(learned, nan=0.0, posinf=0.0, neginf=0.0)
        supervisory = torch.nan_to_num(supervisory, nan=0.0, posinf=0.0, neginf=0.0)

        learned = torch.clamp(learned, min=1e-8)
        supervisory = torch.clamp(supervisory, min=1e-8)

        # Ensure tensors are 2D for sorting/gathering
        if learned.ndim == 1: learned = learned.unsqueeze(0)
        if supervisory.ndim == 1: supervisory = supervisory.unsqueeze(0)

        # Check for empty tensors after potential unsqueezing
        if learned.numel() == 0 or supervisory.numel() == 0:
             fig = go.Figure()
             fig.update_layout(title='Neighbor Selection Probability (Empty Data)')
             return fig

        # Sort based on combined values to handle ties or very similar values
        sort_vals = supervisory + 1e-4 * learned
        indices = torch.argsort(sort_vals, dim=-1, descending=True) # Sort descending for plot X axis later

        probs_sorted_data = self._gather_and_process(learned, indices)
        target_sorted_data = self._gather_and_process(supervisory, indices)

        # Check if gathering produced valid data
        if probs_sorted_data is None or target_sorted_data is None:
             fig = go.Figure()
             fig.update_layout(title='Neighbor Selection Probability (Processing Error)')
             return fig

        probs_mean, _ = probs_sorted_data
        target_mean, _ = target_sorted_data

        # Ensure mean tensors are 1D
        if probs_mean.ndim > 1: probs_mean = probs_mean.squeeze()
        if target_mean.ndim > 1: target_mean = target_mean.squeeze()

        x = np.arange(probs_mean.size(0)) # X axis: 0 to N-1
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=target_mean.numpy(),
            mode='lines+markers', # Changed to lines+markers
            name='Target Distribution P (Mean)',
            marker=dict(color='orange', size = 4), # Slightly larger markers
            line=dict(color='orange', width=1)     # Add line
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=probs_mean.numpy(),
            mode='lines+markers', # Changed to lines+markers
            name='Learned Distribution Q (Mean)',
            marker=dict(color='blue', size = 4), # Slightly larger markers
            line=dict(color='blue', width=1)      # Add line
        ))
        # Optional: Add IQR bands if needed later using fill='tonexty' or error bars
        # _, probs_iqr = probs_sorted_data
        # _, target_iqr = target_sorted_data
        # ... code to add shaded areas ...

        fig.update_layout(
            title='Neighbor Selection Probability Distributions (Mean)',
            xaxis_title='Neighbors Ordered by Target Probability P (High to Low)', # Adjusted title
            yaxis_title='Selection Probability',
            yaxis_type='log',
            template='simple_white',
            legend=dict(
                orientation='h',
                yanchor='bottom', # Anchor legend below plot
                y=1.02,          # Position slightly above plot area
                xanchor='center',
                x=0.5,
                font=dict(size=10),
                itemwidth=30
            ),
            margin=dict(t=80, b=40, l=40, r=20) # Adjust top margin for title/legend
        )

        return fig

    def _gather_and_process(self, kernel: torch.Tensor, indices: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if kernel.numel() == 0 or indices.numel() == 0:
             return None
        # Ensure indices match the dimension being gathered
        if kernel.shape[1] != indices.shape[1]:
             # This might happen if input data had unexpected shapes
             # Attempt to fix or return None
             if kernel.shape[1] < indices.shape[1]: # Too many indices? Trim.
                  indices = indices[:, :kernel.shape[1]]
             else: # Not enough indices? Maybe repeat kernel columns? Less likely needed.
                  return None # Indicate error

        gathered = torch.gather(kernel, 1, indices)

        # Calculate statistics only if there's data after gathering
        if gathered.numel() == 0:
            return None

        mean = gathered.mean(dim=0)

        # Calculate quantiles only if there are enough data points along dim 0
        if gathered.shape[0] >= 4: # Need at least 4 points for robust IQR
             q1, q3 = torch.quantile(gathered, torch.tensor([0.25, 0.75], device=gathered.device), dim=0)
             iqr = q3 - q1
        elif gathered.shape[0] > 0: # Calculate std dev as fallback if few points
             std_dev = gathered.std(dim=0, unbiased=True)
             iqr = std_dev # Use std dev as a measure of spread instead of IQR
        else: # No data points
             iqr = torch.zeros_like(mean)

        # No flipping needed now since we sorted descending initially
        return mean, iqr


class ProbabilitiesStarPlot(BasePlot):
    def plot(self, data: Dict) -> go.Figure:
        probs_tensor = data.get('probabilities')
        labels_tensor = data.get('labels')

        if probs_tensor is None or labels_tensor is None:
             fig = go.Figure()
             fig.update_layout(title='Probabilities Star Plot (Data Missing)')
             return fig

        probs = torch.tensor(probs_tensor)
        labels = torch.tensor(labels_tensor)

        # Handle 1D probabilities (single data point)
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
            # If labels were scalar, make them 1D
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)

        if probs.numel() == 0 or labels.numel() == 0:
             fig = go.Figure()
             fig.update_layout(title='Probabilities Star Plot (Empty Data)')
             return fig

        n_samples, n_clusters = probs.shape

        # Handle case of zero clusters
        if n_clusters == 0:
             fig = go.Figure()
             fig.update_layout(title='Probabilities Star Plot (No Clusters)')
             return fig

        # Get color list
        color_sequence = self.config.cmap.colors if isinstance(self.config.cmap, ListedColormap) else self.config.cmap
        if not isinstance(color_sequence, list): # Fallback
             color_sequence = high_contrast_colors

        theta = torch.linspace(0, 2 * torch.pi, n_clusters + 1)[:-1]
        vertices = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) # Shape: (n_clusters, 2)
        points = probs @ vertices # Shape: (n_samples, 2)

        fig = go.Figure()

        unique_labels = torch.unique(labels)

        # Add cluster points
        for i, label_val in enumerate(unique_labels):
             # Make sure label_val is treated as an integer if possible for indexing colors
             try:
                  label_idx = int(label_val.item())
             except ValueError:
                  label_idx = i # Fallback to sequential index if label isn't numeric

             mask = (labels == label_val)
             if torch.any(mask): # Only add trace if points exist for this label
                  fig.add_trace(go.Scatter(
                       x=points[mask, 0],
                       y=points[mask, 1],
                       mode='markers',
                       name=f'Label {label_val.item()}', # Use actual label value in legend
                       marker=dict(size=6, # Slightly smaller markers
                                   color=color_sequence[label_idx % len(color_sequence)],
                                   opacity=0.5) # Increase transparency
                  ))

        # Add vertices (cluster anchors)
        fig.add_trace(go.Scatter(
             x=vertices[:, 0],
             y=vertices[:, 1],
             mode='markers+text', # Add text labels to vertices
             marker=dict(symbol='circle-open', size=10, color='black', line=dict(width=1)), # Use open circles
             text=[f'C{k}' for k in range(n_clusters)], # Label vertices C0, C1, ...
             textposition="top center",
             textfont=dict(size=10, color='black'),
             name='Cluster Vertices'
        ))

        # Add lines from origin to vertices
        for k in range(n_clusters):
            fig.add_trace(go.Scatter(
                x=[0, vertices[k, 0]],
                y=[0, vertices[k, 1]],
                mode='lines',
                line=dict(color='lightgray', width=1, dash='dot'),
                showlegend=False
            ))


        fig.update_layout(
             title='Probabilities Star Plot (Projected onto Cluster Vertices)',
             xaxis=dict(visible=False, range=[-1.1, 1.1], scaleratio=1), # Maintain aspect ratio
             yaxis=dict(visible=False, range=[-1.1, 1.1], scaleratio=1), # Maintain aspect ratio
             template='simple_white',
             showlegend=True,
             legend_title="Data Labels",
             hovermode='closest', # Improve hover experience
             width=500, # Fixed size
             height=500 # Fixed size
        )
        return fig


class PlotLogger(pl.Callback):
    PLOT_CLASSES = {
        'embeddings': EmbeddingsPlot,
        'cluster_sizes': ClusterSizesPlot,
        'neighborhood_dist': NeighborhoodDistPlot,
        'probabilities_star': ProbabilitiesStarPlot
    }

    def __init__(self, config: Optional[PlotConfig] = None):
        super().__init__()
        self.config = config or PlotConfig()
        # Ensure selected_plots is initialized if None was passed explicitly
        if self.config.selected_plots is None:
             self.config.selected_plots = ['embeddings', 'cluster_sizes']

        self.plots = {
            name: cls(self.config)
            for name, cls in self.PLOT_CLASSES.items()
            # Check if the specific plot type is requested in the config
            if name in self.config.selected_plots
        }
        self._val_outputs = []
        self.epoch_figs = defaultdict(dict)  # {epoch: {plot_name: plotly.Figure}}
        self._widgets_initialized = False
        self._widget_container = None # To hold the VBox display

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Ensure outputs is a dictionary and clone/detach tensors to prevent memory issues
        if isinstance(outputs, dict):
            processed_outputs = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    # Detach from graph and move to CPU to free GPU memory and prevent graph retention
                    processed_outputs[k] = v.detach().cpu()
                else:
                    # Keep non-tensor data as is (e.g., scalar metrics)
                    processed_outputs[k] = v
            self._val_outputs.append(processed_outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._val_outputs or not trainer.logger: # Check if logger exists
            self._val_outputs.clear() # Clear buffer even if not logging
            return

        # Aggregate outputs - handle cases where keys might be missing in some batches
        aggregated_outputs = defaultdict(list)
        all_keys = set(k for batch_out in self._val_outputs for k in batch_out.keys())

        for batch_out in self._val_outputs:
            for k in all_keys:
                value = batch_out.get(k)
                if value is not None:
                    aggregated_outputs[k].append(value)

        # Concatenate tensors, handle non-tensors if necessary
        final_outputs = {}
        plot_keys_needed = {
            'embeddings', 'labels', 'probabilities',
            'learned_distribution', 'supervisory_distribution'
        }
        successful_aggregation = True
        for k in all_keys:
            values = aggregated_outputs[k]
            if not values: continue # Skip if key had no values

            # Check if all items are tensors before trying to concat
            if all(isinstance(v, torch.Tensor) for v in values):
                 # Filter out empty tensors before concat
                 non_empty_values = [v for v in values if v.numel() > 0]
                 if non_empty_values:
                      try:
                           final_outputs[k] = torch.cat(non_empty_values, dim=0)
                      except RuntimeError as e:
                           # Handle concat error (e.g. different shapes)
                           print(f"Warning: Could not concatenate outputs for key '{k}'. Error: {e}")
                           if k in plot_keys_needed: successful_aggregation = False
                 elif k in plot_keys_needed: # Needed key had only empty tensors
                      successful_aggregation = False
                      print(f"Warning: No data aggregated for key '{k}'.")

            elif len(values) == 1: # Keep single non-tensor value if needed
                 final_outputs[k] = values[0]
            # else: handle lists of non-tensors if required by any plot

        # Check if essential data for plots is present
        # 'probabilities' might be the same as 'learned_distribution' depending on model output
        if 'learned_distribution' in final_outputs and 'probabilities' not in final_outputs:
             final_outputs['probabilities'] = final_outputs['learned_distribution']

        required_keys_present = all(k in final_outputs for k in plot_keys_needed if k in aggregated_outputs)

        if not successful_aggregation or not required_keys_present:
            print(f"Epoch {trainer.current_epoch}: Skipping plotting due to missing or incompatible validation outputs.")
            self._val_outputs.clear()
            return

        # Prepare data dictionary specifically for plotting functions
        plot_data = {}
        if 'embeddings' in final_outputs: plot_data['embeddings'] = final_outputs['embeddings'].numpy() # Keep as numpy for plots
        if 'labels' in final_outputs: plot_data['labels'] = final_outputs['labels'].numpy() # Keep as numpy for plots
        # Pass tensors directly where plots handle them (like ClusterSizes, NeighborhoodDist)
        if 'probabilities' in final_outputs: plot_data['probabilities'] = final_outputs['probabilities']
        if 'learned_distribution' in final_outputs: plot_data['learned_distribution'] = final_outputs['learned_distribution']
        if 'supervisory_distribution' in final_outputs: plot_data['supervisory_distribution'] = final_outputs['supervisory_distribution']


        # Filter plot_data based on keys actually needed by selected plots
        active_plot_data = {}
        for name, plot_instance in self.plots.items():
            # Minimal check for required keys for each plot type (can be made more robust)
            keys_needed_for_plot = set()
            if name == 'embeddings': keys_needed_for_plot = {'embeddings', 'labels'}
            elif name == 'cluster_sizes': keys_needed_for_plot = {'probabilities'}
            elif name == 'neighborhood_dist': keys_needed_for_plot = {'learned_distribution', 'supervisory_distribution'}
            elif name == 'probabilities_star': keys_needed_for_plot = {'probabilities', 'labels'}

            # Check if all necessary keys are available in plot_data
            if keys_needed_for_plot.issubset(plot_data.keys()):
                 for k in keys_needed_for_plot:
                      active_plot_data[k] = plot_data[k]
            else:
                print(f"Warning: Skipping plot '{name}' due to missing data keys: {keys_needed_for_plot - set(plot_data.keys())}")
                # Remove plot from self.plots for this epoch if data is missing? Or just skip plotting it.
                # For simplicity, we'll just skip plotting it for now.
                continue # Skip to next plot if data is missing


        # Generate and log plots if data is available
        if active_plot_data:
             self._log_plots(active_plot_data, trainer, trainer.current_epoch)

        self._val_outputs.clear() # Clear buffer for next epoch

    def _log_plots(self, plot_data: Dict, trainer, epoch: int):
        # Generate figures for selected plots
        current_epoch_figs = {}
        for name, plot_instance in self.plots.items():
             # Check again if data is sufficient for this specific plot before plotting
             keys_needed_for_plot = set()
             if name == 'embeddings': keys_needed_for_plot = {'embeddings', 'labels'}
             elif name == 'cluster_sizes': keys_needed_for_plot = {'probabilities'}
             elif name == 'neighborhood_dist': keys_needed_for_plot = {'learned_distribution', 'supervisory_distribution'}
             elif name == 'probabilities_star': keys_needed_for_plot = {'probabilities', 'labels'}

             if not keys_needed_for_plot.issubset(plot_data.keys()):
                 continue # Skip if data missing

             try:
                 fig = plot_instance.plot(plot_data)
                 current_epoch_figs[name] = fig
             except Exception as e:
                 print(f"Error generating plot '{name}' for epoch {epoch}: {e}")
                 # Optionally log traceback: import traceback; traceback.print_exc()


        # Store figures if showing plots interactively
        if self.config.show_plots and current_epoch_figs:
             self.epoch_figs[epoch] = current_epoch_figs
             # Defer widget building/updating until after TensorBoard logging
             # to ensure it happens even if show_plots is False initially but turned on later.

        # Log generated figures to TensorBoard
        for name, fig in current_epoch_figs.items():
             try:
                 img_bytes = pio.to_image(fig, format='png', width=600, height=600, scale=2)
                 image = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
                 trainer.logger.experiment.add_image(
                      tag=f"Plots/{name}", # Group plots in TensorBoard under 'Plots'
                      img_tensor=image,
                      global_step=trainer.global_step, # Use global step for consistency
                      dataformats='HWC'
                 )
             except Exception as e:
                 print(f"Error logging plot '{name}' to TensorBoard for epoch {epoch}: {e}")


        # Update interactive widget *after* logging if enabled
        if self.config.show_plots and current_epoch_figs:
             self._build_or_update_slideshow_widget(epoch)


    def _build_or_update_slideshow_widget(self, current_epoch: int):
         """Builds the widget display if not present, or updates it."""
         # Use display(..., display_id=...) for updatable display in Jupyter
         display_id = 'plot_logger_widget'

         if not self._widgets_initialized:
             self._slider = widgets.IntSlider(
                 value=current_epoch, min=0, max=current_epoch, step=1,
                 description='Epoch:', continuous_update=False, # Update only on release
                 layout=widgets.Layout(width='80%')
             )
             self._play = widgets.Play(
                 value=current_epoch, min=0, max=current_epoch, step=1,
                 interval=1500, description="Play Epochs", disabled=False
             )
             widgets.jslink((self._play, 'value'), (self._slider, 'value'))

             self._output = widgets.Output()
             self._slider.observe(self._update_output_display, names='value')

             # Initial display setup
             self._widget_container = widgets.VBox([
                 widgets.HBox([self._play, self._slider]),
                 self._output
             ])
             display(self._widget_container, display_id=display_id)
             self._widgets_initialized = True
             self._update_output_display() # Initial plot render

         else:
             # Update existing slider/play max value
             self._slider.max = current_epoch
             self._play.max = current_epoch
             # Optionally, automatically move slider to the latest epoch
             # self._slider.value = current_epoch
             # self._play.value = current_epoch
             # No need to call display() again, just update the output area if needed
             # If the slider isn't currently at the max epoch, the user might be inspecting older epochs,
             # so we don't force update the display unless the slider value *changes*.
             # If slider IS at the previous max, update it to the new max epoch plot
             if self._slider.value == current_epoch -1:
                 self._slider.value = current_epoch # This will trigger _update_output_display


    def _update_output_display(self, change=None):
        """Callback function to update the plots displayed in the widget output area."""
        with self._output: # Capture output within the designated area
            clear_output(wait=True) # Clear previous plots
            epoch = self._slider.value
            if epoch in self.epoch_figs and self.epoch_figs[epoch]:
                 # Use the grid display function
                 self._display_figures_grid(
                     list(self.epoch_figs[epoch].values()), # Pass list of figures for the selected epoch
                     columns=2 # Adjust columns as needed
                 )
            else:
                 display(widgets.HTML(value=f"<p>No plots available for epoch {epoch}.</p>"))


    def _display_figures_grid(self, figures: List[go.Figure], columns: int = 2):
        """Renders a list of Plotly figures into an HTML grid."""
        if not figures:
            return

        # Standardize figure size slightly smaller for grid display
        fig_width = 450
        fig_height = 400
        for fig in figures:
             # Ensure layout exists before updating
             if fig.layout is None: fig.layout = go.Layout()
             fig.update_layout(width=fig_width, height=fig_height, margin=dict(l=40, r=20, t=40, b=40))


        # Convert figures to HTML, embedding Plotly.js only in the first one
        html_figs = [
             fig.to_html(
                 include_plotlyjs='cdn' if i == 0 else False,
                 full_html=False,
                 config={'displayModeBar': False} # Optional: hide mode bar for cleaner look
             ) for i, fig in enumerate(figures)
        ]

        # Build HTML table for grid layout
        html = "<div style='display: flex; flex-wrap: wrap; justify-content: space-around;'>"
        for fig_html in html_figs:
            # Each figure in a div for flexbox item styling
            html += f"<div style='margin: 10px;'>{fig_html}</div>"
        html += "</div>"

        display(HTML(html)) # Display the generated HTML grid
