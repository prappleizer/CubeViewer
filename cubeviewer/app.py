import glob
import os
import sys

import astropy.constants as ac
import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from bokeh.events import MouseMove, SelectionGeometry
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    BoxSelectTool,
    Button,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    PreText,
    RadioGroup,
    RangeSlider,
    Select,
    Span,
    TextInput,
    Toggle,
)
from bokeh.palettes import Magma256, Viridis256
from bokeh.plotting import figure


def create_smooth_rdbu_colormap():
    cmap = plt.get_cmap("RdBu")
    return [matplotlib.colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]


RdBu256 = create_smooth_rdbu_colormap()


# --- FITSCube Class Definition ---
class FITSCube:
    def __init__(self, file_path, redshift=0.0, crop_wl=200):
        self.file_path = file_path
        self.redshift = redshift
        self.crop_wl_width = crop_wl
        self.data, self.wcs = self.load_fits_data()
        self.wl_observed = self.compute_wavelengths()[crop_wl:-crop_wl]
        self.wl_rest = self.wl_observed / (1 + self.redshift)
        self.data = self.data[crop_wl:-crop_wl, :, :]
        self.velocity = self.compute_velocity()
        # self.mom0_map = None  # Will store the current moment 0 map

    def set_redshift(self, redshift: float):
        self.redshift = redshift
        self.wl_rest = self.wl_observed / (1 + self.redshift)

    def load_fits_data(self):
        """Loads FITS file and WCS information."""
        data = fits.getdata(self.file_path)
        wcs = WCS(fits.getheader(self.file_path))
        return data, wcs

    def compute_wavelengths(self):
        """Calculates the wavelength array from the FITS data and WCS."""
        indices = np.arange(self.data.shape[0])
        pix_coords = np.zeros((len(indices), 3))
        pix_coords[:, 2] = indices
        world_coords = self.wcs.all_pix2world(pix_coords, 1)
        wavelengths = world_coords[:, 2] * 1e10  # Convert to Angstroms
        return wavelengths

    def get_wavelength_array(self, frame="observed"):
        """Returns the wavelength array in the requested frame."""
        if frame == "rest":
            return self.wl_rest
        return self.wl_observed

    def compute_velocity(
        self,
        offset=0,
    ):
        """Convert wavelengths to velocity relative to a reference wavelength (e.g., the line center)."""
        ref_wl = 6564
        velocity = ((self.wl_observed - ref_wl) / ref_wl) * ac.c.to(u.km / u.s) - (
            offset * (u.km / u.s)
        )
        return velocity.value

    def extract_spectrum(self, ymin, ymax, xmin, xmax, frame="observed"):
        """Extracts a spectrum from the selected region."""
        wl_array = self.get_wavelength_array(frame)
        selected_spectrum = np.nanmean(self.data[:, ymin:ymax, xmin:xmax], axis=(1, 2))
        return wl_array, selected_spectrum

    def moment_0(self, wl_min, wl_max, frame="observed"):
        """Computes the moment 0 map (integrated intensity) between wl_min and wl_max."""
        wl_array = self.get_wavelength_array(frame)
        start_ind = np.where(wl_array > wl_min)[0][0]
        stop_ind = np.where(wl_array > wl_max)[0][0]
        self.mom0_map = np.mean(self.data[start_ind:stop_ind], axis=0)
        return self.mom0_map

    def moment_1(self, wl_min, wl_max, frame="observed"):
        """Computes the moment 1 (velocity field) map."""
        wl_array = self.get_wavelength_array(frame)
        start_ind = np.where(wl_array > wl_min)[0][0]
        stop_ind = np.where(wl_array > wl_max)[0][0]
        intensity = self.data[start_ind:stop_ind]
        velocity = self.compute_velocity(offset=11730)[start_ind:stop_ind]
        intensity_sum = np.sum(intensity, axis=0)
        self.mom1_map = (
            np.sum(velocity[:, None, None] * intensity, axis=0) / intensity_sum
        )
        return self.mom1_map

    def moment_2(self, wl_min, wl_max, frame="observed"):
        """Computes the moment 2 (dispersion) map."""
        wl_array = self.get_wavelength_array(frame)
        start_ind = np.where(wl_array > wl_min)[0][0]
        stop_ind = np.where(wl_array > wl_max)[0][0]
        intensity = self.data[start_ind:stop_ind]
        velocity = self.velocity[start_ind:stop_ind]

        intensity_sum = np.sum(intensity, axis=0)
        velocity_mean = (
            np.sum(velocity[:, None, None] * intensity, axis=0) / intensity_sum
        )
        self.mom2_map = np.sqrt(
            np.sum((velocity[:, None, None] - velocity_mean) ** 2 * intensity, axis=0)
            / intensity_sum
        )
        return self.mom2_map


# --- Application Setup ---

# Get command line arguments for the directory
if "--dir" in sys.argv:
    dir_arg = sys.argv[np.where(np.array(sys.argv) == "--dir")[0][0] + 1]
else:
    dir_arg = "./"
directory = os.path.abspath(dir_arg)
fits_files1 = glob.glob(os.path.join(directory, "*cubew*.fits"))
fits_files2 = glob.glob(os.path.join(directory, "*cubes*.fits"))
fits_files = fits_files1 + fits_files2

fileinput = Select(
    options=[os.path.basename(i) for i in fits_files],
    value=os.path.basename(fits_files[0]),
)


def load_new_fits_file(attr, old, new):
    # Update the cube with the selected file
    cube_path = os.path.join(directory, fileinput.value)  # New file path
    redshift_value = float(redshift_input.value)  # Use current redshift value

    # Reinitialize the FITSCube object with the new file
    global cube
    cube = FITSCube(cube_path, redshift=redshift_value)

    # Update moment zero map and spectrum plot based on the new cube
    min_wl = np.min(cube.wl_observed) + 100
    max_wl = np.max(cube.wl_observed) - 100
    frame = "observed"
    frame_toggle.active = False
    redshift_input.value = "0.0"
    zsetsuccess.text = f"Current Redshift: {cube.redshift}"
    # Update the moment zero map with the newly loaded cube
    moment_zero_map = cube.moment_0(min_wl, max_wl, frame=frame)
    source_image.data = {"image": [moment_zero_map]}

    # Update spectrum to empty or reset it
    source_spectrum.data = {
        "wavelength": cube.get_wavelength_array(frame),
        "spectrum": np.zeros_like(cube.get_wavelength_array(frame)),
    }

    # Recompute color mapper based on the new cube
    min_val, max_val = np.nanpercentile(moment_zero_map, [3, 99.5])
    color_mapper.low = min_val
    color_mapper.high = max_val

    # Update the vertical spans for the moment zero map on the spectrum plot
    vline_min.location = min_wl
    vline_max.location = max_wl

    # Update the plot title to reflect the new file
    p_map.title.text = f"Moment Zero Map: {fileinput.value}"


# Attach the callback to the file input dropdown
fileinput.on_change("value", load_new_fits_file)


redshift_input = TextInput(title="Redshift", value="0.0")

# Initialize the FITSCube with the first file
cube = FITSCube(
    os.path.join(directory, fileinput.value), redshift=float(redshift_input.value)
)

# Create the moment 0 map and plot it
min_wl = np.min(cube.wl_observed) + 100
max_wl = np.max(cube.wl_observed) - 100
moment_zero_map = cube.moment_0(min_wl, max_wl, frame="observed")

# Normalize the image data for color mapping
min_val, max_val = np.nanpercentile(moment_zero_map, [3, 99.5])
color_mapper = LinearColorMapper(palette=Viridis256, low=min_val, high=max_val)

# Set up Bokeh plots
fov_aspect_ratio = 33 / 20.4
p_map = figure(width=800, height=int(800 / fov_aspect_ratio), title="Moment Zero Map")
source_image = ColumnDataSource(data={"image": [moment_zero_map]})
p_map.image(
    image="image",
    source=source_image,
    x=-0.5,
    y=-0.5,
    dw=cube.data.shape[2],
    dh=cube.data.shape[1],
    color_mapper=color_mapper,
)

# Add colorbar to the plot
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0, 0))
p_map.add_layout(color_bar, "right")

# Spectrum plot setup
source_spectrum = ColumnDataSource(
    data={"wavelength": cube.wl_observed, "spectrum": np.zeros_like(cube.wl_observed)}
)
p_spectrum = figure(width=800, height=250, title="Extracted Spectrum")
p_spectrum.line("wavelength", "spectrum", source=source_spectrum)

# Vertical spans for moment 0 map range
vline_min = Span(location=min_wl, dimension="height", line_color="blue", line_width=0.5)
vline_max = Span(location=max_wl, dimension="height", line_color="blue", line_width=0.5)
p_spectrum.add_layout(vline_min)
p_spectrum.add_layout(vline_max)


# Callbacks for frame toggle and wavelength range adjustment
def update_wavelength_axis(attr, old, new):
    # redshift = float(redshift_input.value)
    # cube.redshift = redshift
    frame = "rest" if frame_toggle.active else "observed"

    # Update the spectrum plot
    source_spectrum.data = {
        "wavelength": cube.get_wavelength_array(frame),
        "spectrum": source_spectrum.data["spectrum"],
    }
    frame_toggle.label = (
        "Toggle to Obs Frame" if frame_toggle.active else "Toggle to Rest Frame"
    )
    # Update moment 0 map
    if frame_toggle.active:
        min_wl_input.value = f"{float(min_wl_input.value) / (1 + cube.redshift):.2f}"
        max_wl_input.value = f"{float(max_wl_input.value) / (1 + cube.redshift):.2f}"
    else:
        min_wl_input.value = f"{float(min_wl_input.value) * (1 + cube.redshift):.2f}"
        max_wl_input.value = f"{float(max_wl_input.value) * (1 + cube.redshift):.2f}"
    min_wl = float(min_wl_input.value)
    max_wl = float(max_wl_input.value)
    vline_min.location = min_wl
    vline_max.location = max_wl


def update_moment_map():
    # min_wl = float(min_wl_input.value)
    # max_wl = float(max_wl_input.value)
    # if frame_toggle.active:
    #     min_wl_input.value = f"{float(min_wl_input.value) / (1 + cube.redshift):.2f}"
    #     max_wl_input.value = f"{float(max_wl_input.value) / (1 + cube.redshift):.2f}"
    # else:
    #     min_wl_input.value = f"{float(min_wl_input.value) * (1 + cube.redshift):.2f}"
    #     max_wl_input.value = f"{float(max_wl_input.value) * (1 + cube.redshift):.2f}"
    min_wl = float(min_wl_input.value)
    max_wl = float(max_wl_input.value)
    vline_min.location = min_wl
    vline_max.location = max_wl
    frame = "rest" if frame_toggle.active else "observed"

    if moment_selector.active == 0:  # Moment 0
        moment_map = cube.moment_0(min_wl, max_wl, frame=frame)
        color_mapper.update(palette=Viridis256)
    elif moment_selector.active == 1:  # Moment 1
        moment_map = cube.moment_1(min_wl, max_wl, frame=frame)
        # vmin, vmax = np.nanpercentile(moment_map, [3, 99.5])
        # vmax_abs = max(abs(vmin), abs(vmax))
        # color_mapper.update(palette=RdBu256)  # Diverging palette for velocity
        color_mapper.update(palette=RdBu256, low=-500, high=500)
    elif moment_selector.active == 2:  # Moment 2
        moment_map = cube.moment_2(min_wl, max_wl, frame=frame)
        color_mapper.update(palette=Magma256)

    # Update image data and color bar
    source_image.data = {"image": [moment_map]}
    min_val, max_val = np.nanpercentile(moment_map, [3, 99.5])
    color_mapper.low = min_val
    color_mapper.high = max_val

    # Update plot title
    moment_name = [
        "Moment 0 (Intensity)",
        "Moment 1 (Velocity)",
        "Moment 2 (Dispersion)",
    ][moment_selector.active]
    p_map.title.text = moment_name


# RadioGroup for moment map selection
moment_selector = RadioGroup(labels=["Moment 0", "Moment 1", "Moment 2"], active=0)
moment_selector.on_change("active", lambda attr, old, new: update_moment_map())


# RangeSlider for color mapping (vmin, vmax)
def update_vmin_vmax(attr, old, new):
    vmin, vmax = np.nanpercentile(source_image.data["image"][0], [new[0], new[1]])
    color_mapper.update(low=vmin, high=vmax)


range_slider = RangeSlider(
    start=0,
    end=100,
    value=(3, 99.5),
    step=0.5,
    title="Percentiles",
    sizing_mode="stretch_width",
)
range_slider.on_change("value", update_vmin_vmax)


# BoxSelectTool for selecting a region and extracting spectrum
def extract_spectrum_callback(event):
    geometry = event.geometry
    xmin, xmax = int(geometry["x0"]), int(geometry["x1"])
    ymin, ymax = int(geometry["y0"]), int(geometry["y1"])

    if xmax - xmin < 2:
        xmax += 1
    if ymax - ymin < 2:
        ymax += 1

    frame = "rest" if frame_toggle.active else "observed"
    wl_array, selected_spectrum = cube.extract_spectrum(
        ymin, ymax, xmin, xmax, frame=frame
    )

    # Update spectrum plot
    source_spectrum.data = {
        "wavelength": wl_array,
        "spectrum": selected_spectrum,
    }


p_map.add_tools(BoxSelectTool(persistent=True))
p_map.on_event(SelectionGeometry, extract_spectrum_callback)

# Widgets
min_wl_input = TextInput(value=f"{min_wl:.2f}", title="Min Wavelength (Å)")
max_wl_input = TextInput(value=f"{max_wl:.2f}", title="Max Wavelength (Å)")
set_button = Button(
    label="Set Wavelength Range",
    button_type="success",
    align="end",
)
set_button.on_click(update_moment_map)


# Frame toggle
frame_toggle = Toggle(
    label="Toggle to Rest Frame", button_type="primary", active=False, align="end"
)
frame_toggle.on_change("active", update_wavelength_axis)

# PreText widgets for displaying information
spec_center = PreText(text="")
spec_width = PreText(text="")
window_info = PreText(text="")
window_info2 = PreText(text="")


set_z_button = Button(label="Set Redshift", align="end")


def set_z_callback():
    cube.set_redshift(float(redshift_input.value))
    zsetsuccess.text = f"Current Redshift: {cube.redshift}"


set_z_button.on_click(set_z_callback)

zsetsuccess = PreText(text=f"Current Redshift: {cube.redshift}")


# Create a vertical span that tracks the mouse position
hover_line = Span(
    location=0, dimension="height", line_color="red", line_width=0.7, line_dash="dashed"
)
p_spectrum.add_layout(hover_line)

# Add HoverTool to the spectrum plot (optional)
hover_tool = HoverTool(tooltips=[("Wavelength", "$x{0.2f} Å")], mode="vline")
p_spectrum.add_tools(hover_tool)


# Callback to track the mouse movement over the spectrum panel
def mouse_move_callback(event):
    hover_line.location = event.x  # Update the vertical line position
    window_info.text = f"Mouse at wavelength: {event.x:.2f} Å"


# Attach the callback to mouse movements in the spectrum panel
p_spectrum.on_event(MouseMove, mouse_move_callback)
# Layout
layout = column(
    row(
        column(range_slider, p_map, sizing_mode="stretch_width"),
        column(
            fileinput,
            row(min_wl_input, max_wl_input, set_button),
            row(redshift_input, set_z_button, frame_toggle),
            row(zsetsuccess),
        ),
    ),
    row(
        p_spectrum,
        sizing_mode="stretch_both",
    ),
    sizing_mode="stretch_both",
)
curdoc().add_root(layout)
