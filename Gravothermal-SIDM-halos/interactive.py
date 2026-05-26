"""
That what Mikołaj did - unfortunately not maybe will not work, because I change few settings in arguments of
the plotting function - try to recover.
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QLabel, QComboBox, \
    QHBoxLayout, QRadioButton

import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
# from matplotlib.figure import Figure

from Determining_the_duration_of_evolution import createPhoDataModel, T_from_log_bst, para_F, find_rho_s_r_s, \
    mass_variables, find_paper_parameters
from calculateErrors import relative_difference, find_min_rho_core
from createPlots import onePlotDensityProfileBackend, differenceDensityProfilesBackend
from createPlots_summary import evolution_rho_core_true_backend, maxDifferenceProfilesBackend

import warnings

# Supressing Pandas data loading warning
warnings.filterwarnings('ignore', message='Columns \(1\) have mixed types')
warnings.filterwarnings('ignore', message='divide by zero encountered in log10')

# data folder location
data_dir = 'data'
# regex patterns for folder name and data file name
rho_folder_pat = r'rho_M_(.+)'
rho_file_pat = r'ρsol_M_(\d+\.\d*)_t_(\d+\.\d*)_sigma_(\d+\.\d*)'


class File_struct():
    '''Handles finding datafiles and loading data'''

    def __init__(self, data_dir=data_dir, rho_folder_pat=rho_folder_pat, rho_file_pat=rho_file_pat):
        self.data_dir = data_dir
        self.rho_folder_pat = rho_folder_pat
        self.rho_file_pat = rho_file_pat
        self.file_dict = dict()
        self.__search_data()
        # self.file_dict structure:
        # {mass<str> : {cross_section<float> : path_to_file<str>}}

    def __search_data(self):
        for f in os.scandir(self.data_dir):
            if f.is_dir():
                match = re.match(self.rho_folder_pat, f.name)
                if match:
                    mass = match.group(1)
                    self.file_dict[mass] = dict()

                    for data_file in os.scandir(f):
                        match = re.match(self.rho_file_pat, data_file.name)
                        if match:
                            GALAXYMASS = match.group(1)
                            EVOLUTIONTIME = match.group(2)
                            CROSSSECTION = match.group(3)
                            self.file_dict[mass][float(CROSSSECTION)] = os.path.join(self.data_dir, f.name,
                                                                                     data_file.name)

    def print_struct(self):
        '''Prints out the dirs and files it found'''
        print('Data structure:')
        for key, val in self.file_dict.items():
            print(f'\{key}:')
            for keyf, valf in val.items():
                print(f' |     CS_{float(keyf)} : {valf}')

    def get_masses(self):
        '''Returns the list of found masses (from dir names)'''
        masses = list(self.file_dict.keys())
        masses.sort(key=lambda s: float(s.replace('x10^', 'e').rstrip('0').rstrip('.')))
        return masses

    def get_sigmas(self, mass):
        '''Returns the list of cross sections available for a given mass (as floats)'''
        sigmas = list(self.file_dict[mass])  # [float(x) for x in self.file_dict[mass].keys()]
        sigmas.sort()
        return sigmas

    def get_name(self, mass, sigma):
        '''Returns relative path of a given data file'''
        return self.file_dict[mass][float(sigma)]

    def get_data(self, mass, sigma):
        '''Loads a data file and returns a pandas.dataframe with its data'''
        rho_path = self.get_name(mass, sigma)
        rho_data = pd.read_csv(rho_path, sep='\t', names=['t', 'r', 'rho'])
        rho_data = rho_data.apply(pd.to_numeric, errors='coerce').dropna()  # dropping the lines with broken r
        return rho_data

    def get_model(self, mass, sigma):
        '''Returns the fitted model with given mass and cross section set'''
        rho_data = self.get_data(mass, sigma)
        name = self.get_name(mass, sigma).split('\\')[-1]  # extracting the name to get model params
        match = re.match(self.rho_file_pat, name)
        if match:
            GALAXYMASS = match.group(1)
            EVOLUTIONTIME = match.group(2)
            CROSSSECTION = match.group(3)
            return createPhoDataModel(rho_data['r'], rho_data['t'], float(CROSSSECTION), float(GALAXYMASS))


class MainWindow(QMainWindow):
    '''Qt Window class'''

    def __init__(self, file_struct, init_mass='2x10^11.0', init_sigma=10.0, init_timestep=1000, figsize=(7, 7),
                 plot_dpi=100, output_dpi=300, ENABLE_SAVE=False, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Creating plot object and widget
        fig, ax = plt.subplots(figsize=figsize, dpi=plot_dpi)
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.axes = ax
        self.canvas.fig = fig

        # Creating file structure for handling data files
        self.file_struct = file_struct

        # Initial params
        self.mass = init_mass
        self.sigma = init_sigma
        self.timestep = init_timestep
        self.plot_type = 'density'
        self.output_dpi = output_dpi
        self.ENABLE_SAVE = ENABLE_SAVE

        self.load_data()  # loads the data points from file, using self.mass and self.sigma

        self.update_plot()  # initial plot render

        self.setWindowTitle('Rho plotter')

        # Creating layout
        layout = QVBoxLayout()

        # Radio buttons for plot type
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(QLabel('Plot type:'))

        radio = QRadioButton('Density profile')
        radio.setChecked(True)
        radio.plot_type = 'density'
        radio.toggled.connect(self._plot_type_changed)
        radio_layout.addWidget(radio)

        radio = QRadioButton('Relative error')
        radio.plot_type = 'error'
        radio.toggled.connect(self._plot_type_changed)
        radio_layout.addWidget(radio)

        radio = QRadioButton('Core density evolution')
        radio.plot_type = 'core'
        radio.toggled.connect(self._plot_type_changed)
        radio_layout.addWidget(radio)

        radio = QRadioButton('Max relative error')
        radio.plot_type = 'max_error'
        radio.toggled.connect(self._plot_type_changed)
        radio_layout.addWidget(radio)

        layout.addLayout(radio_layout)

        # Dropdown for mass
        label_mass = QLabel('Mass:')
        layout.addWidget(label_mass)

        dropdown_mass = QComboBox()
        dropdown_mass.addItems(self.file_struct.get_masses())
        dropdown_mass.setMaxVisibleItems(len(self.file_struct.get_masses()))
        dropdown_mass.setCurrentText(self.mass)
        dropdown_mass.currentTextChanged.connect(self._mass_changed)
        layout.addWidget(dropdown_mass)

        # Dropdown for cross section
        label_cs = QLabel('Cross section:')
        layout.addWidget(label_cs)

        self.dropdown_cs = QComboBox()
        # self.dropdown_cs.addItems([str(x) for x in self.file_struct.get_sigmas(self.mass)])
        # self.dropdown_cs.setMaxVisibleItems(len(self.file_struct.get_sigmas(self.mass)))
        # self.dropdown_cs.setCurrentText(str(self.sigma))
        self._update_sigma()
        self.dropdown_cs.currentTextChanged.connect(self._sigma_changed)
        layout.addWidget(self.dropdown_cs)

        # Adding plot to layout
        layout.addWidget(self.canvas)

        # Timestep slider
        self.label_slider = QLabel('Time step:')
        self.label_slider.setText(f'Time step:{self.timestep}')
        layout.addWidget(self.label_slider)

        self.slider = QSlider(Qt.Horizontal)

        self.slider.setMinimum(0)
        self.slider.setMaximum(self.time_steps - 1)
        self.slider.setValue(self.timestep)  # setting initial value
        self.slider.setSingleStep(1)

        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.sliderReleased.connect(self._update_save_btns)

        layout.addWidget(self.slider)

        # testbtn = QPushButton('Test')
        # testbtn.pressed.connect(lambda: self.label_slider.setEnabled(not self.label_slider.isEnabled()))
        # layout.addWidget(testbtn)

        # Save buttons
        if self.ENABLE_SAVE:
            save_layout = QHBoxLayout()

            self.save_btn_pdf = QPushButton('pdf')
            self.save_btn_pdf.pressed.connect(lambda: self.save_plot('.pdf'))
            save_layout.addWidget(self.save_btn_pdf)

            self.save_btn_png = QPushButton('png')
            self.save_btn_png.pressed.connect(lambda: self.save_plot('.png'))
            save_layout.addWidget(self.save_btn_png)

            self._update_save_btns()

            layout.addLayout(save_layout)

        # Bilding the main layout
        widget = QWidget()
        widget.setLayout(layout)

        # Set the central widget of the Window.
        self.setCentralWidget(widget)

        # Display the window
        self.show()

    def _slider_moved(self, i):
        '''Updates plot when timestep slider is moved'''
        self.timestep = i
        self.label_slider.setText(f'Time step:{self.timestep}')
        self.update_plot()
        # self._update_save_btns() # Update is on `sliderReleased`, not on `valueChanged`

    def _mass_changed(self, s):
        '''Updates plot when mass is changed'''
        self.mass = s
        self.load_data()
        self._update_sigma()
        self.update_plot()
        self._update_save_btns()

    def _sigma_changed(self, s):
        '''Updates plot when cross section is changed'''
        self.sigma = float(s)
        self.load_data()
        self.update_plot()
        self._update_save_btns()

    def _plot_type_changed(self):
        '''Updates plot when plot type is changed'''
        radio = self.sender()
        self.plot_type = radio.plot_type
        self.update_plot()
        self._update_save_btns()
        if self.plot_type in ('density', 'error'):
            self.slider.setEnabled(True)
            self.label_slider.setEnabled(True)
        else:
            self.slider.setEnabled(False)
            self.label_slider.setEnabled(False)

    def _update_sigma(self):
        '''Updates the cross section dropdown list'''
        self.dropdown_cs.blockSignals(True)

        self.dropdown_cs.clear()
        self.dropdown_cs.addItems([str(x) for x in self.file_struct.get_sigmas(self.mass)])
        self.dropdown_cs.setMaxVisibleItems(len(self.file_struct.get_sigmas(self.mass)))
        if self.sigma in self.file_struct.get_sigmas(self.mass):
            self.dropdown_cs.setCurrentText(str(self.sigma))
        else:
            self.dropdown_cs.setCurrentIndex(0)
            self.sigma = float(self.dropdown_cs.currentText())

        self.dropdown_cs.blockSignals(False)

    def update_plot(self):
        '''Redraws the plot'''
        ax = self.canvas.axes
        ax.cla()
        fig = self.canvas.fig

        # find parameters of paper model
        GALAXYMASS, EVOLUTIONTIME, CROSSSECTION = self.extract_galaxy_params_from_filename()
        _power_mass = float(GALAXYMASS)
        _sigma_m = float(CROSSSECTION)

        #  ------------------------------------ Actual plotting ------------------------------------
        Title_name = f'Mass: {("{:.1e}".format(10 ** float(GALAXYMASS)))} [M_s], sigma: {("{:.1e}".format(float(CROSSSECTION)))} [cm^2/g],\nt: {("{:.1e}".format(float(EVOLUTIONTIME)))} [Gyr]'

        if self.plot_type == 'density':
            # Plotting procedure
            onePlotDensityProfileBackend(fig, ax, _power_mass, _sigma_m,
                                         self.rho_data['t'], self.rho_data['r'], self.rho_data['rho'],
                                         self.model_rho,
                                         self.timestep, '')
        elif self.plot_type == 'error':
            # Plotting procedure
            differenceDensityProfilesBackend(fig, ax, _power_mass, _sigma_m,
                                             self.rho_data['t'], self.rho_data['r'], self.rho_data['rho'],
                                             self.model_rho,
                                             self.timestep, '')
        elif self.plot_type == 'core':
            evolution_rho_core_true_backend(fig, ax, self.rho_data['t'], self.rho_data['r'], self.rho_data['rho'],
                                            self.model_rho,
                                            Title_name,
                                            10)
        elif self.plot_type == 'max_error':
            core_simulation = find_min_rho_core(self.rho_data['t'], self.rho_data['r'], self.rho_data['rho'], 10)
            core_fitting = find_min_rho_core(self.rho_data['t'], self.rho_data['r'], self.model_rho, 10)
            rho_s_and_r_s = find_rho_s_r_s(_power_mass, mass_variables)
            rho_s = rho_s_and_r_s[0]
            r_s = rho_s_and_r_s[1]
            # calculating time
            T_change = T_from_log_bst(para_F, _sigma_m, rho_s, r_s)
            maxDifferenceProfilesBackend(fig, ax, self.rho_data['t'], self.rho_data['rho'],
                                         self.model_rho,
                                         Title_name,
                                         core_simulation[1], core_fitting[1], T_change)
        else:
            ax.text(0.5, 0.5, 'Not yet implemented!', fontsize=30, color='red', ha='center')
        self.canvas.draw()

    def load_data(self):
        '''Loads data from file (selected based on current self.mass and self.sigma)'''
        if self.sigma not in self.file_struct.get_sigmas(self.mass):  # Making sure, that self.sigma is available
            self._update_sigma()
        file_name = self.file_struct.get_name(self.mass, self.sigma)
        print(f'Loading data from {file_name}...')
        self.rho_data = self.file_struct.get_data(self.mass, self.sigma)
        self.model_rho = self.file_struct.get_model(self.mass, self.sigma)

        # different values of time. In another words number of steps in time
        len_t = 0
        while self.rho_data['t'][len_t] == self.rho_data['t'][0]: len_t += 1

        self.len_t = len_t

        self.time_steps = int(len(self.rho_data['t']) / len_t)
        print(f'Finished loading {file_name}!')

    def extract_galaxy_params_from_filename(self):
        '''Extracts GALAXYMASS, EVOLUTIONTIME and CROSSSECTION corresponding to current self.mass and self.sigma
            returns GALAXYMASS, EVOLUTIONTIME, CROSSSECTION'''
        name = os.path.basename(self.file_struct.get_name(self.mass, self.sigma))
        match = re.match(self.file_struct.rho_file_pat, name)
        if match:
            GALAXYMASS = match.group(1)
            EVOLUTIONTIME = match.group(2)
            CROSSSECTION = match.group(3)
            return GALAXYMASS, EVOLUTIONTIME, CROSSSECTION

    def save_plot(self, extension):
        print(f'Saving file {self.get_output_file_name(extension)}...')
        save_loc = './Plots/' + self.get_output_file_name(extension)
        GUI_size = self.canvas.fig.get_size_inches()
        fig_sizes = {'density': (7.0, 7.0), 'error': (9.0, 7.0), 'core': (15.0, 10.0), 'max_error': (9.0, 7.0)}
        self.canvas.fig.set_size_inches(*fig_sizes.get(self.plot_type, (7.0, 7.0)))
        self.canvas.fig.savefig(save_loc, dpi=self.output_dpi)
        print(f'Saved!')
        self.canvas.fig.set_size_inches(*GUI_size)
        self.update_plot()

    def _update_save_btns(self):
        if hasattr(self, 'save_btn_pdf') and hasattr(self, 'save_btn_png'):
            self.save_btn_pdf.setText(f'Save as\n{self.get_output_file_name(".pdf")}')
            self.save_btn_png.setText(f'Save as\n{self.get_output_file_name(".png")}')

    def get_output_file_name(self, extension):
        base = {'density': 'density_profile', 'error': 'relative_difference', 'core': 'rho_core',
                'max_error': 'max_relative_difference'}
        mass = f"{float(self.mass.replace('x10^', 'e').rstrip('0').rstrip('.')):-.1e}".replace('+',
                                                                                               '')  # self.mass.rstrip('0').rstrip('.')#
        if self.plot_type in ('density', 'error'):
            time_point = self.rho_data['t'][self.timestep * self.len_t + 1]
        elif self.plot_type in ('core', 'max_error'):
            time_point = self.rho_data['t'][(self.time_steps - 1) * self.len_t + 1]
        time = "%10.2e" % (10 ** time_point)
        time = time.replace(' ', '')
        return f'{base.get(self.plot_type, "")}_m_{mass}_sigma_{self.sigma}_t_{time}_Gyr{extension}'


if __name__ == '__main__':
    file_struct = File_struct()
    # print(file_struct.get_masses())
    # print(file_struct.get_sigmas('2x10^11.0'))
    # print(file_struct.get_name('2x10^11.0', 10.0))
    # print(file_struct.get_model('2x10^11.0', 10.0))
    file_struct.print_struct()

    app = QApplication([])

    window = MainWindow(file_struct, ENABLE_SAVE=False)
    window.show()

    app.exec()