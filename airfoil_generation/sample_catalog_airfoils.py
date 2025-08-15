import matplotlib.pyplot as plt
import numpy as np
import tarfile
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy import interpolate
import random
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)


class AirfoilsDataset(Dataset):
    """Airfoils dataset."""

    def __init__(self, tar_file, num_airfoils=0):
        """
        Args:
            tar_file (string): Path to the the tar file which contains all of the airfoils
        """
        self.airfoils_tar = tarfile.open(tar_file, 'r')
        self.airfoils = self.airfoils_tar.getmembers()
        self.num_airfoils = num_airfoils
        self.data_len = len(self.airfoils_tar.getmembers())

    def __len__(self):
        return max(len(self.airfoils_tar.getmembers()), self.num_airfoils)

    def __getitem__(self, idx):
        idx = idx % self.data_len

        f = self.airfoils_tar.extractfile(self.airfoils[idx])
        # extract the airfoil coords in a dataframe
        df = pd.read_table(f, sep='\s+', header=0, names=['x', 'y'], encoding='ISO-8859-1',
                           index_col=False, on_bad_lines='skip').drop_duplicates()
        x = pd.to_numeric(df.x, errors='coerce').dropna()
        y = pd.to_numeric(df.y, errors='coerce').dropna()
        sample = self.interp_to_common_num_points(np.array([x.to_numpy(), y.to_numpy()]).transpose())

        return sample, self.get_airfoil_name(idx)

    def get_airfoil_name(self, idx):
        return self.airfoils_tar.getnames()[idx]

    def get_airfoil_idx(self, name):
        return self.airfoils_tar.getnames().index(name)

    def interp_to_common_num_points(self, airfoil_coords):
        # create a spline on the front of the airfoil
        airfoil_front = airfoil_coords[airfoil_coords[:, 0] <= 0.4]
        t = np.linspace(0.0, 1.0, 600)
        front_spline = interpolate.splprep([airfoil_front[:, 0], airfoil_front[:, 1]], s=0.00007, k=3)
        front_spline = np.array(interpolate.splev(t, front_spline[0], der=0)).transpose()

        # interpolate the top and bottom sides
        le_idx = np.linalg.norm(airfoil_coords, axis=1).argmin()
        side_1 = np.flip(airfoil_coords[:le_idx, :], axis=0)
        side_1 = side_1[side_1[:, 0] > 0.4]
        side_2 = airfoil_coords[le_idx:, :]
        side_2 = side_2[side_2[:, 0] > 0.4]
        interp_points_1 = np.linspace(side_1[0,0], side_1[-1,0], 200)
        interp_points_2 = np.linspace(side_2[0,0], side_2[-1,0], 200)
        side_1 = np.interp(interp_points_1, side_1[:, 0], side_1[:, 1])
        side_2 = np.interp(interp_points_2, side_2[:, 0], side_2[:, 1])

        # concat the 3 parts to form airfoil
        coo = np.concatenate((np.flip(np.array([interp_points_1, side_1]).T, axis=0), front_spline, np.array([interp_points_2, side_2]).T), axis=0)

        # test to detect crossing trailing edge, if so redraw a sample
        # le_idx = np.linalg.norm(coo, axis=1).argmin()
        # top_side = np.flip(coo[:le_idx, :], axis=0)
        # bot_side = coo[le_idx:, :]
        # interp_points = np.linspace(0.2, 1.0, 500)
        # top_points = np.interp(interp_points, top_side[:, 0], top_side[:, 1])
        # bot_points = np.interp(interp_points, bot_side[:, 0], bot_side[:, 1])
        # if np.less(top_points, bot_points).any():
        #     coo, _ = self.__getitem__(random.randint(0, self.data_len))

        return coo

def sample_catalog_airfoils(catalog_path='clean_airfoil_database_v3.tar', num_airfoils=10, plot_shapes=False, previously_simulated_airfoils=set()):
    airfoils_dataset = AirfoilsDataset(catalog_path)
    if num_airfoils is None:
        num_airfoils = len(airfoils_dataset)

    airfoils = []
    names = []
    
    # Calculate the number of available unsimulated airfoils
    available_airfoils = len(airfoils_dataset) - len(previously_simulated_airfoils)
    num_airfoils = min(num_airfoils, available_airfoils)

    if num_airfoils == 0:
        print("No unsimulated airfoils left.")
        return [], []
    data_loader = DataLoader(airfoils_dataset, batch_size=len(airfoils_dataset), shuffle=True)
    all_airfoils, all_names = next(iter(data_loader))
        
    for airfoil, name in zip(all_airfoils, all_names):
        if name not in previously_simulated_airfoils:
                airfoils.append(airfoil)
                names.append(name)
                if len(airfoils) == num_airfoils:
                    break

    if plot_shapes:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        for shape in airfoils:
            ax.plot(shape[:, 0], shape[:, 1], '.')
        ax.set_aspect('equal')
        plt.show()

    return list(airfoils), list(names)

def sample_specific_airfoils(catalog_path='clean_airfoil_database_v3.tar', airfoil_names=[], plot_shapes=False):
    airfoils_dataset = AirfoilsDataset(catalog_path)
    airfoils = [airfoils_dataset[airfoils_dataset.get_airfoil_idx(name)][0] for name in airfoil_names]
    names = [airfoils_dataset[airfoils_dataset.get_airfoil_idx(name)][1] for name in airfoil_names]

    if plot_shapes:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        for shape in airfoils:
            ax.plot(shape[:, 0], shape[:, 1])
        ax.set_aspect('equal')
        plt.show()

    return airfoils, names

if __name__ == "__main__":
    previously_simulated_airfoils = ['a18.dat']
    sample_catalog_airfoils(catalog_path='clean_mini_database.tar', num_airfoils=2, plot_shapes = True, previously_simulated_airfoils=previously_simulated_airfoils)
    # sample_specific_airfoils(airfoil_names=['vr13.dat'], plot_shapes=True)