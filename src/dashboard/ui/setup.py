# src/dashboard/ui/setup.py

import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
from typing import Dict, Tuple, Any


def display_terminal_config() -> Dict[str, Any]:
    """
    Renders the Streamlit UI for all terminal configuration parameters.

    This includes general settings like power and frequency, array dimensions
    and spacing, and the terminal's position and orientation in 3D space.

    Returns:
        A flat dictionary containing all the configured terminal parameters
        from the UI widgets.
    """
    config = {}

    with st.expander("Terminal Configuration", expanded=True):
        st.subheader("General Properties")
        c1, c2 = st.columns(2)
        config['tx_power'] = c1.number_input(
            "Tx Power (dBm)", -20.0, 30.0, 0.0, step=0.1,
            help="Transmit power in dBm[cite: 68]."
        )
        config['frequency'] = c2.number_input(
            "Frequency (GHz)", 2.0, 300.0, 28.0, step=0.1,
            help="Carrier frequency of the signal in GHz[cite: 69]."
        )

        st.subheader("Transmitter (Tx) Array")
        c1, c2 = st.columns(2)
        config['tx_size_x'] = c1.number_input(
            "Tx Array Size X", 1, 128, 1, help="Number of antennas along the x-axis[cite: 69, 77]."
        )
        config['tx_size_y'] = c2.number_input(
            "Tx Array Size Y", 1, 128, 1, help="Number of antennas along the y-axis[cite: 69, 77]."
        )

        if st.toggle("Advanced Tx Spacing ↗", key="tx_spacing_toggle", help="Define custom spacing for Tx antenna elements instead of the default half-wavelength."):
            sc1, sc2 = st.columns(2)
            config['tx_spacing_x'] = sc1.number_input(
                "Tx Spacing X (factor of λ)", 0.1, 50.0, 0.5, step=0.05, format="%.2f",
                help="Spacing along the x-axis as a multiple of the wavelength (λ)[cite: 73, 82]."
            )
            config['tx_spacing_y'] = sc2.number_input(
                "Tx Spacing Y (factor of λ)", 0.1, 50.0, 0.5, step=0.05, format="%.2f",
                help="Spacing along the y-axis as a multiple of the wavelength (λ)[cite: 73, 82]."
            )
        else:
            config['tx_spacing_x'] = config['tx_spacing_y'] = None

        st.subheader("Receiver (Rx) Array")
        c1, c2 = st.columns(2)
        config['rx_size_x'] = c1.number_input(
            "Rx Array Size X", 1, 128, 16, help="Number of antennas along the x-axis[cite: 70, 77]."
        )
        config['rx_size_y'] = c2.number_input(
            "Rx Array Size Y", 1, 128, 16, help="Number of antennas along the y-axis[cite: 70, 77]."
        )

        if st.toggle("Advanced Rx Spacing ↗", key="rx_spacing_toggle", value=True, help="Define custom spacing for Rx antenna elements instead of the default half-wavelength."):
            sc1, sc2 = st.columns(2)
            config['rx_spacing_x'] = sc1.number_input(
                "Rx Spacing X (factor of λ)", 0.1, 50.0, 10.0, step=0.05, format="%.2f",
                help="Spacing along the x-axis as a multiple of the wavelength (λ)[cite: 74, 82]."
            )
            config['rx_spacing_y'] = sc2.number_input(
                "Rx Spacing Y (factor of λ)", 0.1, 50.0, 10.0, step=0.05, format="%.2f",
                help="Spacing along the y-axis as a multiple of the wavelength (λ)[cite: 74, 82]."
            )
        else:
            config['rx_spacing_x'] = config['rx_spacing_y'] = None

    with st.expander("Terminal Position and Orientation", expanded=True):
        st.subheader("Base Position (m)")
        c1, c2, c3 = st.columns(3)
        config['terminal_x'] = c1.number_input("X", -10., 10., 0., 0.01, format="%.2f", help="Base position of the terminal in meters[cite: 70].")
        config['terminal_y'] = c2.number_input("Y", -10., 10., 0., 0.01, format="%.2f", help="Base position of the terminal in meters[cite: 70].")
        config['terminal_z'] = c3.number_input("Z", 0., 10., 0., 0.01, format="%.2f", help="Base position of the terminal in meters[cite: 70].")

        st.subheader("Tx/Rx Array Offsets from Base (m)")
        c1, c2, c3 = st.columns(3)
        config['tx_offset_x'] = c1.number_input("Tx Offset X", -1., 1., 0., 0.01, help="Offset of the Tx array center relative to the terminal base[cite: 71].")
        config['tx_offset_y'] = c2.number_input("Tx Offset Y", -1., 1., 0., 0.01, help="Offset of the Tx array center relative to the terminal base[cite: 71].")
        config['tx_offset_z'] = c3.number_input("Tx Offset Z", -1., 1., 0., 0.01, help="Offset of the Tx array center relative to the terminal base[cite: 71].")

        c1, c2, c3 = st.columns(3)
        config['rx_offset_x'] = c1.number_input("Rx Offset X", -1., 1., 0., 0.01, help="Offset of the Rx array center relative to the terminal base[cite: 72].")
        config['rx_offset_y'] = c2.number_input("Rx Offset Y", -1., 1., 0., 0.01, help="Offset of the Rx array center relative to the terminal base[cite: 72].")
        config['rx_offset_z'] = c3.number_input("Rx Offset Z", -1., 1., 0., 0.01, help="Offset of the Rx array center relative to the terminal base[cite: 72].")

        st.subheader("Orientation (degrees)")
        c1, c2, c3 = st.columns(3)
        config['elevation'] = c1.number_input(
            "Elevation (θ)", -90.0, 90.0, 0.0, step=0.1, help="Rotation around X-axis. Must be between -90 and 90 degrees[cite: 71, 78]."
        )
        config['azimuth'] = c2.number_input(
            "Azimuth (φ)", -180.0, 180.0, 0.0, step=0.1, help="Rotation around Z-axis. Must be between -180 and 180 degrees[cite: 71, 78]."
        )
        config['tilt'] = c3.number_input(
            "Tilt (τ)", -90.0, 90.0, 0.0, step=0.1, help="Rotation around Y-axis. Must be between -90 and 90 degrees[cite: 71, 79]."
        )
    return config


def display_target_config() -> Tuple[str, Dict[str, Any]]:
    """
    Renders the Streamlit UI for all target configuration parameters.

    This includes selecting the target type (e.g., points, shapes, file),
    and specifying its placement and properties.

    Returns:
        A tuple containing:
        - target_type (str): The selected category of the target.
        - target_params (dict): A dictionary of the target's parameters.
    """
    with st.expander("Target Configuration", expanded=True):
        target_mode = st.radio(
            "Target Type",
            ["Points", "2D Shapes", "Upload File", "Sketch"],
            index=1,
            horizontal=True,
            help="Select the type of target to simulate[cite: 42]."
        )

        st.markdown("###### Target Placement")
        c1, c2, c3 = st.columns(3)
        center_x = c1.number_input("Center X (m)", -20.0, 20.0, 0.0, step=0.1)
        center_y = c2.number_input("Center Y (m)", -20.0, 20.0, 0.0, step=0.1)
        center_z = c3.number_input("Center Z (m)", 0.0, 20.0, 0.5, step=0.1)
        center = (center_x, center_y, center_z)
        target_params = {'center': center}

        if target_mode == "Points":
            target_type = "points"
            pt_str = st.text_input(
                "Point(s) relative to center", "0,0,0",
                help="Enter one or more points as 'x,y,z', separated by semicolons ';'."
            )
            points_list = []
            try:
                for p_s in pt_str.split(';'):
                    x, y, z = map(float, p_s.strip().split(','))
                    points_list.append((center_x + x, center_y + y, center_z + z))
                target_params['positions'] = points_list
            except ValueError:
                st.error("Invalid point format. Use 'x,y,z' separated by semicolons.")
                target_params['positions'] = [center]

        elif target_mode == "2D Shapes":
            target_type = "2d_shapes"
            shape_options = ["Circle", "Ellipse", "Rectangle", "Regular Polygon", "Star", "Cross", "N Random Shapes"]
            shape_type = st.selectbox("Shape", shape_options, index=shape_options.index("Regular Polygon"))
            target_params.update({"shape": shape_type.lower().replace(" ", "_")})

            if shape_type == "N Random Shapes":
                target_params["n_shapes"] = st.number_input(
                    "Number of Shapes (N)", min_value=1, max_value=50, value=5,
                    help="Number of random shapes to generate within the scene[cite: 62]."
                )
            else:
                target_params["rotation"] = st.slider("Rotation (deg)", 0, 360, 0)
                if shape_type == "Circle":
                    target_params["radius"] = st.number_input("Radius (m)", 0.1, 10.0, 0.5, step=0.1)
                elif shape_type == "Ellipse":
                    target_params["major_axis"] = st.number_input("Major Axis (m)", 0.1, 10.0, 1.0, step=0.1)
                    target_params["minor_axis"] = st.number_input("Minor Axis (m)", 0.1, 10.0, 0.5, step=0.1)
                elif shape_type == "Rectangle":
                    target_params["width"] = st.number_input("Width (m)", 0.1, 10.0, 1.0, step=0.1)
                    target_params["height"] = st.number_input("Height (m)", 0.1, 10.0, 0.5, step=0.1)
                elif shape_type == "Regular Polygon":
                    target_params["n_sides"] = st.number_input(
                        "Number of Edges", 3, 20, 3, help="Set the number of sides for the polygon[cite: 56]."
                    )
                    target_params["side_length"] = st.number_input("Side Length (m)", 0.1, 10.0, 1.0, step=0.1)
                elif shape_type == "Star":
                    target_params["points"] = st.number_input("Number of Spikes", 3, 20, 5, help="Set the number of points for the star[cite: 57].")
                    target_params["outer_radius"] = st.number_input("Outer Radius (m)", 0.1, 10.0, 1.0, step=0.1)
                    target_params["inner_radius"] = st.number_input("Inner Radius (m)", 0.1, 10.0, 0.5, step=0.1)
                elif shape_type == "Cross":
                    target_params["arm_length"] = st.number_input("Arm Length (m)", 0.1, 10.0, 1.0, step=0.1)
                    target_params["arm_width"] = st.number_input("Arm Width (m)", 0.05, 5.0, 0.2, step=0.05)

        elif target_mode in ["Upload File", "Sketch"]:
            target_type = "upload_file" if target_mode == "Upload File" else "sketch"
            target_params['scale'] = st.number_input(
                "Image Scale (m)", 0.1, 50.0, 2.0, step=0.1,
                help="Physical width and height of the image in the scene."
            )
            if target_type == "upload_file":
                uploaded_file = st.file_uploader("Upload target image")
                if uploaded_file:
                    img = Image.open(uploaded_file).convert('L')
                    # Invert image so black lines have high reflectivity
                    target_params['img_array'] = 1.0 - (np.array(img).astype(np.float32) / 255.0)
            else:  # Sketch
                canvas = st_canvas(
                    fill_color="black", stroke_width=10, background_color="white",
                    height=200, width=200, drawing_mode="freedraw", key="canvas"
                )
                if canvas.image_data is not None:
                    img_gray = cv2.cvtColor(canvas.image_data.astype(np.uint8), cv2.COLOR_BGRA2GRAY)
                    # Drawn lines are black (value 0), so invert to make them reflective
                    target_params['img_array'] = (img_gray < 128).astype(np.float32)

            if 'img_array' in target_params:
                st.image(target_params['img_array'], caption="Target Reflectivity", use_container_width=True)

    return target_type, target_params




