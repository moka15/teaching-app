from shiny import reactive, req
from shiny.express import ui, render, input
import numpy as np
import plotly.graph_objects as go
from scipy.special import lpmv, factorial

# UI Components
ui.page_opts(title="Spherical Harmonics Visualization", fillable=False)

# Constants and server-side calculations
N = 128  # Resolution (reduced for performance)

# Initial values for parameters
INITIAL_L = 2
INITIAL_M = 1

with ui.sidebar():
    ui.h3("Spherical Harmonics")
    ui.p("Use the sliders below to select specific (l, m) values for visualization.")
    ui.input_slider("l_value", "l value", min=0, max=5, value=INITIAL_L)
    # Set initial max based on l value
    ui.input_slider("m_value", "m value", min=0, max=INITIAL_L, value=INITIAL_M, step=1)
    ui.hr()
    ui.h4("Selected Mode")


    @render.text
    def selected_info():
        l, m = input.l_value(), input.m_value()
        return f"Selected: l = {l}, m = {m}"

ui.h2("Spherical Harmonics Visualization")


@reactive.calc
def compute_harmonics():
    # Generate grid
    t = np.linspace(np.pi, 0, N)  # theta: 180° to 0°
    p = np.linspace(-np.pi, np.pi, N)  # phi: -180° to 180°

    T, P = np.meshgrid(t, p, indexing='ij')

    l = input.l_value()
    m = input.m_value()

    # Calculate Legendre polynomial
    if l == 0:
        Plm = np.ones_like(T)
    else:
        # Using scipy's lpmv function (note: it takes |m|)
        Plm = lpmv(m, l, np.cos(T))

    # Calculate theta component
    prefactor = (-1) ** ((m + abs(m)) / 2) * np.sqrt(
        (2 * l + 1) * factorial(l - abs(m)) / (4 * np.pi * factorial(l + abs(m))))
    theta_component = prefactor * Plm / ((-1) ** m)

    # Calculate phi component
    phi_component = np.exp(1j * m * P)

    return {
        't': t,
        'p': p,
        'theta_values': theta_component,
        'phi_values': phi_component
    }


# Update the m_value slider range based on the selected l_value
@reactive.effect
@reactive.event(input.l_value)
def update_m_range():
    l_val = input.l_value()
    m_val = min(input.m_value(), l_val)
    ui.update_slider("m_value", max=l_val, value=m_val)


with ui.layout_columns():
    # Theta plot (amplitude)
    with ui.card():
        @render.ui
        def theta_plot():
            harmonics = compute_harmonics()
            l, m = input.l_value(), input.m_value()

            # Create sphere with amplitude values
            theta_vals = harmonics['theta_values']

            # Generate sphere coordinates
            phi = np.linspace(0, 2 * np.pi, N)
            theta = np.linspace(0, np.pi, N)
            PHI, THETA = np.meshgrid(phi, theta)

            # Convert to Cartesian coordinates
            X = np.sin(THETA) * np.cos(PHI)
            Y = np.sin(THETA) * np.sin(PHI)
            Z = np.cos(THETA)

            # Create surface plot
            fig = go.Figure()

            fig.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    surfacecolor=theta_vals,
                    colorscale='RdBu',
                    colorbar=dict(
                        title="Amplitude",
                        orientation="h",  # Make colorbar horizontal
                        y=1.0,  # Place colorbar at the top
                        yanchor="bottom",  # Anchor at bottom of colorbar
                        x=0.5,  # Center colorbar horizontally
                        xanchor="center",  # Anchor at center of colorbar
                        len=0.6,  # Make colorbar 60% of the plot width
                        thickness=20  # Make colorbar slightly thicker
                    ),
                    cmin=-np.max(np.abs(theta_vals)),
                    cmax=np.max(np.abs(theta_vals))
                )
            )

            fig.update_layout(
                title=f"Amplitude Θ<sub>{l}</sub><sup>{m}</sup>(θ)",
                scene=dict(
                    aspectmode='cube',
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="z"
                ),
                margin=dict(l=0, r=0, t=80, b=0),  # Increased top margin for colorbar
                paper_bgcolor="white",
                height=500
            )

            # Convert the Plotly figure to HTML
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    # Phi plot (phase)
    with ui.card():
        @render.ui
        def phi_plot():
            harmonics = compute_harmonics()
            l, m = input.l_value(), input.m_value()

            # Create sphere with phase values
            phi_vals = np.angle(harmonics['phi_values'])

            # Generate sphere coordinates
            phi = np.linspace(0, 2 * np.pi, N)
            theta = np.linspace(0, np.pi, N)
            PHI, THETA = np.meshgrid(phi, theta)

            # Convert to Cartesian coordinates
            X = np.sin(THETA) * np.cos(PHI)
            Y = np.sin(THETA) * np.sin(PHI)
            Z = np.cos(THETA)

            # Create surface plot
            fig = go.Figure()

            fig.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    surfacecolor=phi_vals,
                    colorscale='twilight',
                    colorbar=dict(
                        title="Phase",
                        orientation="h",  # Make colorbar horizontal
                        y=1.0,  # Place colorbar at the top
                        yanchor="bottom",  # Anchor at bottom of colorbar
                        x=0.5,  # Center colorbar horizontally
                        xanchor="center",  # Anchor at center of colorbar
                        len=0.6,  # Make colorbar 60% of the plot width
                        thickness=20  # Make colorbar slightly thicker
                    ),
                    cmin=-np.pi,
                    cmax=np.pi
                )
            )

            fig.update_layout(
                title=f"Phase Φ<sub>{m}</sub>(φ)",
                scene=dict(
                    aspectmode='cube',
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="z"
                ),
                margin=dict(l=0, r=0, t=80, b=0),  # Increased top margin for colorbar
                paper_bgcolor="white",
                height=500
            )

            # Convert the Plotly figure to HTML
            return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
