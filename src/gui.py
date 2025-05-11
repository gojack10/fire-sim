import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QGridLayout, QGroupBox, QFileDialog
from PyQt6.QtCore import Qt, QTimer
# We will need matplotlib for plotting, ensure it's installed and import its PyQt6 backend
import matplotlib
matplotlib.use('QtAgg') # Use QtAgg backend for PyQt6
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import simulation logic (assuming it's in the same directory or src is in PYTHONPATH)
# For now, let's assume we will import specific functions as needed
from . import simulation
from . import data_prep
import numpy as np # For placeholder data

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wildfire Simulation Control")
        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._create_controls_panel()
        self._create_simulation_display_panel()

        # Simulation and data state
        self.dem_file_path = None
        self.elevation_data = None # Will hold the loaded elevation numpy array
        self.simulation_grid = None
        self.simulation_history = []
        self.current_step = 0
        self.simulation_timer = QTimer(self)
        self.simulation_timer.timeout.connect(self._simulation_step_triggered)
        self.timer_interval_ms = 500 # Default timer interval
        self.ignition_points = [] # Initialize ignition points

    def _create_controls_panel(self):
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_panel.setFixedWidth(400)

        # --- Simulation Parameters --- 
        sim_params_group = QGroupBox("Simulation Parameters")
        sim_params_layout = QGridLayout()

        sim_params_layout.addWidget(QLabel("Grid Rows:"), 0, 0)
        self.grid_rows_input = QLineEdit("50")
        sim_params_layout.addWidget(self.grid_rows_input, 0, 1)

        sim_params_layout.addWidget(QLabel("Grid Cols:"), 1, 0)
        self.grid_cols_input = QLineEdit("50")
        sim_params_layout.addWidget(self.grid_cols_input, 1, 1)

        sim_params_layout.addWidget(QLabel("Fuel Load (kg/m^2):"), 2, 0)
        self.fuel_load_input = QLineEdit("1.0")
        sim_params_layout.addWidget(self.fuel_load_input, 2, 1)

        sim_params_layout.addWidget(QLabel("Moisture Content (%):"), 3, 0)
        self.moisture_input = QLineEdit("0.2") # e.g. 0.0 to 1.0
        sim_params_layout.addWidget(self.moisture_input, 3, 1)

        sim_params_layout.addWidget(QLabel("Cell Resolution (m):"), 4, 0)
        self.cell_resolution_input = QLineEdit("30.0")
        sim_params_layout.addWidget(self.cell_resolution_input, 4, 1)

        sim_params_layout.addWidget(QLabel("Max Steps:"), 5, 0)
        self.max_steps_input = QLineEdit("100")
        sim_params_layout.addWidget(self.max_steps_input, 5, 1)

        sim_params_layout.addWidget(QLabel("Moisture Threshold:"), 6, 0)
        self.moisture_threshold_input = QLineEdit("0.3")
        sim_params_layout.addWidget(self.moisture_threshold_input, 6, 1)

        sim_params_layout.addWidget(QLabel("Wind Speed (m/s):"), 7, 0)
        self.wind_speed_input = QLineEdit("0.0")
        sim_params_layout.addWidget(self.wind_speed_input, 7, 1)

        sim_params_layout.addWidget(QLabel("Wind Direction (deg):"), 8, 0)
        self.wind_direction_input = QLineEdit("0.0")
        sim_params_layout.addWidget(self.wind_direction_input, 8, 1)

        sim_params_layout.addWidget(QLabel("Base Spread Prob:"), 9, 0)
        self.base_prob_input = QLineEdit("0.58") # From simulation.py default
        sim_params_layout.addWidget(self.base_prob_input, 9, 1)

        sim_params_layout.addWidget(QLabel("Wind Strength Factor:"), 10, 0)
        self.wind_strength_input = QLineEdit("0.5") # From simulation.py default
        sim_params_layout.addWidget(self.wind_strength_input, 10, 1)

        sim_params_layout.addWidget(QLabel("Slope Coefficient:"), 11, 0)
        self.slope_coefficient_input = QLineEdit("0.1") # From simulation.py default
        sim_params_layout.addWidget(self.slope_coefficient_input, 11, 1)

        sim_params_group.setLayout(sim_params_layout)
        controls_layout.addWidget(sim_params_group)

        # --- Terrain Configuration --- 
        terrain_group = QGroupBox("Terrain Configuration")
        terrain_layout = QVBoxLayout()
        self.load_dem_button = QPushButton("Load DEM File")
        self.load_dem_button.clicked.connect(self._load_dem)
        terrain_layout.addWidget(self.load_dem_button)
        self.random_terrain_button = QPushButton("Use Random Terrain")
        self.random_terrain_button.clicked.connect(self._use_random_terrain)
        terrain_layout.addWidget(self.random_terrain_button)
        # Terrain preview placeholder (could be another FigureCanvas)
        self.terrain_figure = Figure(figsize=(5, 3))
        self.terrain_preview_canvas = FigureCanvas(self.terrain_figure)
        terrain_layout.addWidget(self.terrain_preview_canvas)
        self.terrain_preview_ax = self.terrain_figure.add_subplot(111)
        self.terrain_preview_ax.set_title("Terrain Preview / Ignition", fontsize=9)
        self.terrain_preview_ax.set_xticks([])
        self.terrain_preview_ax.set_yticks([])

        # Connect click event for ignition points
        self.terrain_preview_canvas.mpl_connect('button_press_event', self._on_terrain_click)

        terrain_group.setLayout(terrain_layout)
        controls_layout.addWidget(terrain_group)

        # --- Simulation Controls --- 
        sim_controls_group = QGroupBox("Simulation Controls")
        sim_controls_layout = QHBoxLayout()
        self.initialize_button = QPushButton("Initialize Grid")
        self.initialize_button.clicked.connect(self._initialize_simulation_grid)
        self.initialize_button.setEnabled(False) # Disable by default
        sim_controls_layout.addWidget(self.initialize_button)
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_simulation)
        self.start_button.setEnabled(False)
        sim_controls_layout.addWidget(self.start_button)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._pause_simulation)
        self.pause_button.setEnabled(False)
        sim_controls_layout.addWidget(self.pause_button)
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self._step_simulation)
        self.step_button.setEnabled(False)
        sim_controls_layout.addWidget(self.step_button)
        self.reset_button = QPushButton("Reset Sim")
        self.reset_button.clicked.connect(self._reset_simulation)
        self.reset_button.setEnabled(False)
        sim_controls_layout.addWidget(self.reset_button)
        sim_controls_group.setLayout(sim_controls_layout)
        controls_layout.addWidget(sim_controls_group)

        controls_layout.addStretch()
        self.main_layout.addWidget(controls_panel)

    def _create_simulation_display_panel(self):
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        # self._plot_initial_grid() # Placeholder for initial plot

        display_layout.addWidget(self.canvas)
        self.main_layout.addWidget(display_panel)

    def _load_dem(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open DEM File",
            ".",  # Start directory
            "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            print(f"Selected DEM file: {file_path}")
            # Store this path or load data directly
            # For now, just store it as an example
            self.dem_file_path = file_path
            try:
                self.elevation_data = data_prep.load_dem(self.dem_file_path)
                print(f"DEM data loaded successfully. Shape: {self.elevation_data.shape}")
                self.statusBar().showMessage(f"Loaded DEM: {self.dem_file_path}")
                self.initialize_button.setEnabled(True) # Enable after loading terrain
                self._update_terrain_preview() # Call this once preview is implemented
            except Exception as e:
                self.elevation_data = None # Clear any old data
                self.dem_file_path = None
                self.ignition_points = [] # Clear ignition points
                self.simulation_grid = None # Clear old simulation grid
                self.initialize_button.setEnabled(False)
                self.start_button.setEnabled(False)
                self.pause_button.setEnabled(False)
                self.step_button.setEnabled(False)
                self.reset_button.setEnabled(False)
                print(f"Error loading DEM file: {e}")
                self.statusBar().showMessage(f"Error loading DEM file: {e}")
            
            # Next step: load the actual DEM data using this path
            # self.elevation_data = data_prep.load_dem(file_path)
            # Or, initialize part of the grid:
            # temp_grid = simulation.initialize_grid((1,1), elevation_source=file_path) # Dummy size for now
            # self.elevation_data = temp_grid['elevation']
            # print("DEM data loaded (placeholder)")
            # self._update_terrain_preview() # Call this once preview is implemented
            self.statusBar().showMessage(f"DEM file selected: {file_path}")
        else:
            self.statusBar().showMessage("DEM file selection cancelled.")

    def _use_random_terrain(self):
        print("Use Random Terrain clicked")
        try:
            rows = int(self.grid_rows_input.text())
            cols = int(self.grid_cols_input.text())
            if rows <= 0 or cols <= 0:
                raise ValueError("Grid dimensions must be positive.")
            
            # Using initialize_grid to get a full grid structure, then extracting elevation
            # This is consistent with how initialize_grid is used for DEMs too.
            # We can also pass other default parameters for fuel, moisture if needed, 
            # or get them from other UI elements later.
            self.elevation_data = simulation.generate_random_terrain(rows, cols)
            self.dem_file_path = "Random Terrain"
            self._update_terrain_preview()
            self.statusBar().showMessage(f"Generated random terrain ({rows}x{cols}).")
            self.initialize_button.setEnabled(True) # Enable after generating terrain
            # Clear any previous ignition points if we generate new terrain
            self.ignition_points = []
        except ValueError as ve:
            self.statusBar().showMessage(f"Invalid grid dimensions: {ve}")
            # QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid grid dimensions: {ve}")
        except Exception as e:
            self.statusBar().showMessage(f"Error generating random terrain: {e}")
            print(f"Error generating random terrain: {e}")
            # QtWidgets.QMessageBox.warning(self, "Error", f"Could not generate random terrain: {e}")

    def _update_terrain_preview(self):
        self.terrain_preview_ax.clear()
        self.ax.clear() # Also clear the main simulation display axes

        if self.elevation_data is not None:
            # Update terrain preview
            self.terrain_preview_ax.imshow(self.elevation_data, cmap='terrain')
            for r, c in self.ignition_points:
                self.terrain_preview_ax.plot(c, r, 'rx', markersize=5)
            self.terrain_preview_ax.set_xticks([])
            self.terrain_preview_ax.set_yticks([])
            self.terrain_preview_canvas.draw()

            # Update main simulation display (initially same as preview)
            self.ax.imshow(self.elevation_data, cmap='terrain')
            for r, c in self.ignition_points:
                self.ax.plot(c, r, 'rx', markersize=8) # Slightly larger markers for main display
            self.ax.set_title("Simulation Grid (Click on preview to set ignition)", fontsize=10)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
        else:
            # Clear terrain preview
            self.terrain_preview_ax.text(0.5, 0.5, 'No Terrain Data', horizontalalignment='center', verticalalignment='center')
            self.terrain_preview_ax.set_xticks([])
            self.terrain_preview_ax.set_yticks([])
            self.terrain_preview_canvas.draw()

            # Clear main simulation display
            self.ax.text(0.5, 0.5, 'Load/Generate Terrain and Set Ignition Points', horizontalalignment='center', verticalalignment='center', wrap=True)
            self.ax.set_title("Simulation Grid", fontsize=10)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()

    def _get_simulation_parameters(self):
        params = {}
        try:
            params['grid_rows'] = int(self.grid_rows_input.text())
            params['grid_cols'] = int(self.grid_cols_input.text())
            if params['grid_rows'] <= 0 or params['grid_cols'] <= 0:
                raise ValueError("Grid dimensions must be positive.")

            params['fuel_load'] = float(self.fuel_load_input.text())
            if params['fuel_load'] < 0:
                raise ValueError("Fuel load cannot be negative.")

            params['moisture'] = float(self.moisture_input.text())
            if not (0.0 <= params['moisture'] <= 1.0):
                raise ValueError("Moisture must be between 0.0 and 1.0.")

            params['cell_resolution'] = float(self.cell_resolution_input.text())
            if params['cell_resolution'] <= 0:
                raise ValueError("Cell resolution must be positive.")

            params['max_steps'] = int(self.max_steps_input.text())
            if params['max_steps'] <= 0:
                raise ValueError("Max steps must be positive.")

            params['moisture_threshold'] = float(self.moisture_threshold_input.text())
            if not (0.0 <= params['moisture_threshold'] <= 1.0):
                raise ValueError("Moisture threshold must be between 0.0 and 1.0.")

            params['wind_speed'] = float(self.wind_speed_input.text())
            # Wind speed can be 0 or positive
            if params['wind_speed'] < 0:
                raise ValueError("Wind speed cannot be negative.")

            params['wind_direction'] = float(self.wind_direction_input.text())
            # Wind direction typically 0-360, can be normalized later by simulation logic
            # For now, no strict validation beyond being a float

            params['base_prob'] = float(self.base_prob_input.text())
            if not (0.0 <= params['base_prob'] <= 1.0):
                raise ValueError("Base spread probability must be between 0.0 and 1.0.")

            params['wind_strength'] = float(self.wind_strength_input.text())
            # Wind strength can be 0 or positive
            if params['wind_strength'] < 0:
                raise ValueError("Wind strength factor cannot be negative.")

            params['slope_coefficient'] = float(self.slope_coefficient_input.text())
            # Slope coefficient can be positive, negative or zero.

            self.statusBar().showMessage("Parameters successfully parsed.")
            return params
        except ValueError as ve:
            self.statusBar().showMessage(f"Invalid parameter input: {ve}")
            # print(f"Invalid parameter input: {ve}") # For debugging
            # QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid parameter: {ve}")
            return None

    def _on_terrain_click(self, event):
        if event.inaxes != self.terrain_preview_ax or self.elevation_data is None:
            return
    
        # matplotlib's event.xdata and event.ydata are float indices
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        # Check if click is within bounds of the data array
        if 0 <= row < self.elevation_data.shape[0] and 0 <= col < self.elevation_data.shape[1]:
            if (row, col) not in self.ignition_points:
                self.ignition_points.append((row, col))
            else:
                self.ignition_points.remove((row, col))
            self._update_terrain_preview()

    def _initialize_simulation_grid(self):
        print("Initialize Simulation Grid clicked")

        if self.elevation_data is None:
            self.statusBar().showMessage("Error: Terrain data not loaded. Please load a DEM or generate random terrain first.")
            # Ensure dependent buttons remain disabled or are explicitly disabled
            self.start_button.setEnabled(False)
            self.step_button.setEnabled(False)
            self.reset_button.setEnabled(False) # Or handle reset state appropriately
            return

        if not self.ignition_points:
            self.statusBar().showMessage("Error: No ignition points set. Click on the terrain preview to set ignition points.")
            # Ensure dependent buttons remain disabled
            self.start_button.setEnabled(False)
            self.step_button.setEnabled(False)
            self.reset_button.setEnabled(False) # Or handle reset state appropriately
            return

        params = self._get_simulation_parameters()
        if params is None:
            self.start_button.setEnabled(False)
            self.step_button.setEnabled(False)
            return

        try:
            # Initialize the grid structure first
            self.simulation_grid = simulation.initialize_grid(
                size=(params['grid_rows'], params['grid_cols']),
                fuel_load=params['fuel_load'],
                moisture=params['moisture'],
                elevation_source=self.elevation_data, # Pass the actual elevation data array
                cell_resolution=params['cell_resolution']
            )

            # Now, apply ignition points to the newly created grid
            if self.simulation_grid is not None and self.ignition_points:
                if self.elevation_data is not None and self.elevation_data.shape[0] > 0 and self.elevation_data.shape[1] > 0:
                    scale_r = params['grid_rows'] / self.elevation_data.shape[0]
                    scale_c = params['grid_cols'] / self.elevation_data.shape[1]

                    for r_orig, c_orig in self.ignition_points:
                        # Scale the coordinates
                        r_scaled = int(round(r_orig * scale_r))
                        c_scaled = int(round(c_orig * scale_c))

                        # Ensure scaled ignition points are within the new grid dimensions
                        if 0 <= r_scaled < params['grid_rows'] and 0 <= c_scaled < params['grid_cols']:
                            simulation.ignite(self.simulation_grid, r_scaled, c_scaled, time=0)
                            # print(f"DEBUG: Ignited ({r_scaled},{c_scaled}). State: {self.simulation_grid['state'][r_scaled, c_scaled]}. CellState.BURNING is {simulation.CellState.BURNING}")
                        else:
                            print(f"Warning: Scaled ignition point ({r_scaled}, {c_scaled}) from original ({r_orig}, {c_orig}) is outside configured grid dimensions and was skipped.")
                else: 
                    self.statusBar().showMessage("Error: Cannot scale ignition points, elevation data is missing or invalid.", 5000)
            
            # print("DEBUG: Ignition loop finished or skipped.")
            # if self.simulation_grid is not None and self.ignition_points and self.elevation_data is not None:
            #     if len(self.ignition_points) > 0:
            #         r_orig_last, c_orig_last = self.ignition_points[-1]
            #         scale_r_last = params['grid_rows'] / self.elevation_data.shape[0]
            #         scale_c_last = params['grid_cols'] / self.elevation_data.shape[1]
            #         r_s_last = int(round(r_orig_last * scale_r_last))
            #         c_s_last = int(round(c_orig_last * scale_c_last))
            #         if 0 <= r_s_last < params['grid_rows'] and 0 <= c_s_last < params['grid_cols']:
            #             print(f"DEBUG: After loop, state of last processed ignition point ({r_s_last},{c_s_last}): {self.simulation_grid['state'][r_s_last, c_s_last]}")

            self.current_step = 0 
            # print(f"DEBUG: self.current_step set to {self.current_step}.")

            self.ax.clear()
            # print("DEBUG: self.ax.clear() called.")

            if self.simulation_grid is None:
                print("CRITICAL ERROR: self.simulation_grid is None before imshow in _initialize_simulation_grid. Initialization failed.")
                self.statusBar().showMessage("Critical Error: Simulation grid is None after init attempt.")
                return # Cannot proceed
            elif 'state' not in self.simulation_grid.dtype.names:
                print(f"CRITICAL ERROR: 'state' field missing in simulation_grid. Dtype: {self.simulation_grid.dtype}")
                self.statusBar().showMessage("Critical Error: Grid 'state' field missing.")
                return # Cannot proceed
            
            try:
                # Corrected to use CellState.BURNED
                self.ax.imshow(self.simulation_grid['state'], cmap='hot', vmin=int(simulation.CellState.UNBURNED), vmax=int(simulation.CellState.BURNED))
                # print("DEBUG: self.ax.imshow() successfully called.")
            except Exception as imshow_exception:
                print(f"ERROR during self.ax.imshow() in _initialize_simulation_grid: {type(imshow_exception).__name__} - {str(imshow_exception)}")
                import traceback
                traceback.print_exc()
                raise 

            self.ax.set_title(f"Initial Grid State - Step {self.current_step}")
            # print("DEBUG: self.ax.set_title() successfully called.")

            self.canvas.draw()
            # print("DEBUG: self.canvas.draw() successfully called.")

            self.statusBar().showMessage("Simulation grid initialized successfully with ignition points.")
            # print("DEBUG: Success status message set.")
            
            # self._update_main_simulation_display() # Call this once implemented
            # self.ax.clear() # Already cleared above

            self.initialize_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False) # Pause only makes sense when running
            self.step_button.setEnabled(True)
            self.reset_button.setEnabled(True)

        except Exception as e:
            self.simulation_grid = None
            self.statusBar().showMessage(f"Error initializing grid: {e}")
            print(f"Error initializing grid: {e}")
            # QtWidgets.QMessageBox.warning(self, "Grid Init Error", f"Could not initialize grid: {e}")

    def _reset_simulation(self):
        print("Reset Simulation clicked")
        self.simulation_grid = None
        self.current_step = 0
        self.ignition_points = [] 
        if hasattr(self, 'simulation_timer') and self.simulation_timer.isActive():
            self.simulation_timer.stop()

        self.elevation_data = None 
        self.dem_file_path = None
        self.initialize_button.setEnabled(False) # Disable if terrain is cleared

        self._update_terrain_preview() # Clear displays
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Grid Reset. Load/Generate Terrain and Initialize Grid again.', horizontalalignment='center', verticalalignment='center', wrap=True)
        self.ax.set_title("Simulation Grid", fontsize=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

        self.initialize_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.step_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.statusBar().showMessage("Simulation reset. Load terrain and initialize grid.")

    def _simulation_step_triggered(self):
        params = self._get_simulation_parameters()
        if params is None:
            # Error already handled by _get_simulation_parameters
            return False # Indicate failure

        if self.simulation_grid is None:
            self.statusBar().showMessage("Error: Simulation grid not initialized.")
            return False # Indicate failure

        max_steps = params.get('max_steps', 100) # Default if not found, though it should be there

        if self.current_step >= max_steps:
            self.statusBar().showMessage(f"Max steps ({max_steps}) reached.")
            # Potentially stop timer here if running
            if hasattr(self, 'simulation_timer') and self.simulation_timer.isActive():
                self.simulation_timer.stop()
            self.start_button.setEnabled(False) # Cannot continue if max steps reached
            self.pause_button.setEnabled(False)
            self.step_button.setEnabled(False)
            return False # Indicate simulation cannot proceed

        try:
            # print(f"Step {self.current_step}: Running simulation step...")
            # print(f"  Wind Speed: {params['wind_speed']}, Wind Dir: {params['wind_direction']}")
            # print(f"  Base Prob: {params['base_prob']}, Wind Str: {params['wind_strength']}, Slope Coeff: {params['slope_coefficient']}")
            # print(f"  Moisture Thresh: {params['moisture_threshold']}")
            
            step_params = {k: v for k, v in params.items() if k not in ['grid_rows', 'grid_cols', 'max_steps']}
            self.simulation_grid = simulation.step(
                self.simulation_grid,
                time=self.current_step,
                **step_params
            )
            self.current_step += 1

            # Update main display
            self.ax.clear()
            # Corrected to use CellState.BURNED
            self.ax.imshow(self.simulation_grid['state'], cmap='hot', vmin=int(simulation.CellState.UNBURNED), vmax=int(simulation.CellState.BURNED))
            # Overlay elevation with transparency for context
            self.ax.imshow(self.simulation_grid['elevation'], cmap='Greys', alpha=0.3, vmin=np.min(self.simulation_grid['elevation']), vmax=np.max(self.simulation_grid['elevation']))
            self.ax.set_title(f"Simulation Step: {self.current_step}")
            self.canvas.draw()
            self.statusBar().showMessage(f"Simulation step: {self.current_step}")

            # Check for end condition (no more burning cells)
            if not np.any(self.simulation_grid['state'] == simulation.CellState.BURNING):
                self.statusBar().showMessage(f"Simulation ended at step {self.current_step}: No more burning cells.")
                if hasattr(self, 'simulation_timer') and self.simulation_timer.isActive():
                    self.simulation_timer.stop()
                self.start_button.setEnabled(False) # Cannot continue if ended
                self.pause_button.setEnabled(False)
                self.step_button.setEnabled(False)
                return False # Indicate simulation ended
            
            return True # Indicate step was successful and can continue

        except Exception as e:
            self.statusBar().showMessage(f"Error during simulation step: {e}")
            print(f"Error during simulation step {self.current_step}: {e}")
            # QtWidgets.QMessageBox.warning(self, "Simulation Error", f"Error at step {self.current_step}: {e}")
            if hasattr(self, 'simulation_timer') and self.simulation_timer.isActive():
                self.simulation_timer.stop()
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.step_button.setEnabled(False)
            return False # Indicate failure

    def _step_simulation(self):
        print("Step Simulation clicked")
        if self.simulation_grid is None:
            self.statusBar().showMessage("Initialize grid before stepping.")
            return
        
        can_continue = self._simulation_step_triggered()
        # Button states are handled within _simulation_step_triggered or when max_steps/end_condition is met
        if not can_continue:
            print("Simulation cannot continue or has ended.")
            # Further actions if needed, e.g. if timer was running for _start_simulation

    def _start_simulation(self):
        print("Start Simulation clicked")
        if self.simulation_grid is None:
            self.statusBar().showMessage("Initialize grid before starting.")
            return
        
        # Ensure parameters are still valid, though they shouldn't change post-init without re-init
        params = self._get_simulation_parameters()
        if params is None:
            self.statusBar().showMessage("Cannot start: Invalid simulation parameters.")
            return
        
        max_steps = params.get('max_steps', 100)
        if self.current_step >= max_steps:
            self.statusBar().showMessage(f"Cannot start: Max steps ({max_steps}) already reached.")
            return
        if not np.any(self.simulation_grid['state'] == simulation.CellState.BURNING) and self.current_step > 0:
             # Check if any burning cells exist if simulation has already started once (current_step > 0)
            self.statusBar().showMessage(f"Cannot start: Simulation already ended at step {self.current_step}.")
            return

        self.start_button.setText("Start") # Ensure text is 'Start', not 'Resume'
        # Optionally, get timer_interval_ms from a UI element here
        self.simulation_timer.start(self.timer_interval_ms)
        self.statusBar().showMessage(f"Simulation started. Step: {self.current_step}")

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.step_button.setEnabled(False)
        self.initialize_button.setEnabled(False) # Cannot re-initialize while running
        # Reset Sim should remain enabled

    def _pause_simulation(self):
        print("Pause Simulation clicked")
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.statusBar().showMessage(f"Simulation paused at step {self.current_step}.")

            self.start_button.setEnabled(True) # Becomes 'Resume'
            self.start_button.setText("Resume")
            self.pause_button.setEnabled(False)
            self.step_button.setEnabled(True) # Allow stepping while paused
        else:
            self.statusBar().showMessage("Simulation is not running.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Ensure src is in path if running gui.py directly for development
    import os
    if 'src' not in os.getcwd(): # A bit simplistic, better to manage PYTHONPATH or run as module
        # This is a common pattern but might need adjustment based on project structure
        # For now, we assume 'python -m src.gui' or proper PYTHONPATH setup for imports like 'from . import simulation'
        pass 

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
