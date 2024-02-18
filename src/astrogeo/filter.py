import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from astrogeo.image import Image
from astrogeo.consts import CENTER, CMAP


class Filter(object):
	def __init__(self, df: pd.DataFrame, ratio: int = 10) -> None:
		self.maps = df
		self.dirty_maps, self.weird_maps = None, None
		self.filtered_maps = None
		self.filter_df(ratio)
	
	def draw_sn_dist(self) -> None:
		test_dir = 'src/astrogeo/test'
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		
		signal_noise = [
			sig / nl 
			for sig, nl in zip(self.maps.map_max, self.maps.noise_level)
		]
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_title('Signal/noise distribution')
		sns.histplot(signal_noise)
		ax.set_xlabel('Signal-noise ratio')
		ax.set_ylabel('Number')
		plt.savefig(f'{test_dir}/signal_noise_dist.png', dpi=500)
		plt.close(fig)

	def find_weird_maps(self) -> set:
		# FIXME: change pixel difference
		df = self.maps
		weird_maps = [
			file for c_x, x, c_y, y, a, file in
			zip(df.mapc_x, df.map_max_x, df.mapc_y, 
	   		df.map_max_y, df.obs_author, df.file_name)
			if (abs(c_x - x) > 3 or abs(c_y - y) > 3) and a != 'Alan Marscher'
		]
		return set(weird_maps)
	
	def uv_data_len(self) -> set:
		...
	
	def maps_w_bad_signal_noise(self, ratio: int) -> dict:
		df = self.maps
		signal_noise = {
			file: sig / nl for file, sig, nl in 
			zip(df.file_name, df.map_max, df.noise_level)
			if sig/nl <= ratio
		}
		return signal_noise
	
	def filter_df(self, ratio: int) -> None:
		self.weird_maps = self.find_weird_maps()
		self.dirty_maps = self.maps_w_bad_signal_noise(ratio)
		self.filtered_maps = self.weird_maps.union(self.dirty_maps)

		for x in self.maps.index:
			b_maj, b_min = self.maps.loc[x, 'b_maj'], self.maps.loc[x, 'b_min']
			file_name = self.maps.loc[x, 'file_name']
			pixel_size = self.maps.loc[x, 'pixel_size_y']
			if (b_maj == -1 or b_min == -1 or 
	   			file_name in self.filtered_maps or
				b_maj * 3.6e6 / pixel_size > 60):
				self.maps.drop(x, inplace=True)
	
	def _draw_map(self, ax: np.array, path: str, file_name: str) -> Image:
		dir = file_name.split('_')[0]
		im = Image(f'{path}/{dir}/{file_name}')
		data = im.map_data().squeeze()
		ax.imshow(data, cmap=CMAP, origin='lower')
		ax.set_title(im.object, loc=CENTER)
		return im
	
	def draw_dirty_maps(self, path: str, file_name: str) -> None:
		step = 4
		n = len(self.dirty_maps.keys())
		left = n % step
		files = sorted(list(self.dirty_maps.keys()))
		with PdfPages(f'{file_name}.pdf') as pdf:
			for i in range(0, n-step, step):
				fig, axes = plt.subplots(2, 2)
				self._draw_map(axes[0][0], path, files[i])
				self._draw_map(axes[0][1], path, files[i+1])
				self._draw_map(axes[1][0], path, files[i+2])
				self._draw_map(axes[1][1], path, files[i+3])
				fig.tight_layout()
				pdf.savefig()
				plt.close(fig)
			
			for i in range(left, 0, -1):
				fig, ax = plt.subplots(1)
				self._draw_map(ax, path, files[-i])
				fig.tight_layout()
				pdf.savefig()
				plt.close(fig)
	
	def _draw_map_w_author(
			self, ax: np.array, path: str, file_name: str
		) -> None:
		im = self._draw_map(ax, path, file_name)
		ax.set_title(f'{im.object}_{im.author}', loc=CENTER)
	
	def draw_filtered_maps(self, path: str, file_name: str) -> None:
		step = 4
		maps = self.filtered_maps
		n = len(maps)
		left = n % step
		files = sorted(list(maps))
		with PdfPages(f'{file_name}.pdf') as pdf:
			for i in range(0, n-step, step):
				fig, axes = plt.subplots(2, 2)
				self._draw_map_w_author(axes[0][0], path, files[i])
				self._draw_map_w_author(axes[0][1], path, files[i+1])
				self._draw_map_w_author(axes[1][0], path, files[i+2])
				self._draw_map_w_author(axes[1][1], path, files[i+3])
				fig.tight_layout()
				pdf.savefig()
				plt.close(fig)
			
			for i in range(left, 0, -1):
				fig, ax = plt.subplots(1)
				self._draw_map_w_author(ax, path, files[-i])
				fig.tight_layout()
				pdf.savefig()
				plt.close(fig)

class BeamCluster(Filter):
	test_dir = 'src/astrogeo/test'
	def __init__(self, df: pd.DataFrame, ratio: int = 10) -> None:
		Filter.__init__(self, df, ratio)
		self.kmeans, self.X = None, None
		self.b_maj, self.b_min = None, None
		if not os.path.exists(self.test_dir):
			os.makedirs(self.test_dir)
	
	def _preprocess(self) -> np.array:
		pixel_size = self.maps['pixel_size_y'].to_numpy().T
		self.b_maj = self.maps['b_maj'].to_numpy().T * 3.6e6 / pixel_size
		self.b_min = self.maps['b_min'].to_numpy().T * 3.6e6 / pixel_size
		self.b_pa = self.maps['b_pa'].to_numpy().T

		return np.stack([self.b_maj, self.b_min, self.b_pa])
	
	def _beam_clustering(self, clusters: int) -> None:
		X = self._preprocess()
		kms = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
		self.kmeans = kms.fit(X.T)
		labels = np.array([self.kmeans.labels_])
		self.X = np.concatenate((X.T, labels.T), axis=1)

	def beam_cluster_means(self, clusters: int) -> pd.DataFrame:
		if self.X is None:
			self._beam_clustering(clusters)
		
		data = self.X
		means = {}
		for ind, label in enumerate(data[:, 3]):
			label = int(label)
			if label not in means:
				means[label] = [data[ind, 0], data[ind, 1], data[ind, 2], 1]
			else:
				for i in range(3):
					means[label][i] += data[ind, i]
				means[label][3] += 1
		
		for label in means:
			count = means[label][3]
			for i in range(3):
				means[label][i] /= count
		
		df = pd.DataFrame(means).T
		df = df.rename(
			columns={0: 'b_maj', 1: 'b_min', 2: 'b_pa', 3: 'amount'}
		)
		df.index.name = 'Cluster'
		df.amount = df.amount.astype(int)
		for col in df.columns:
			if col != 'amount':
				df[col] = df[col].apply(
					lambda x: np.format_float_scientific(x, precision=2)
				)
		df = df.astype(float)
		return df.sort_index()
		
	def draw_beam_clustering(self, clusters: int) -> None:
		if self.kmeans is None or self.X is None:
			self._beam_clustering(clusters)
		X = self.X
		test_dir = self.test_dir
		
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.scatter(
			X[:, 0], X[:, 1], X[:, 2], 
			c=self.kmeans.labels_.astype(float)
		)
		ax.set_xlabel('Beam MAJ [pixels]')
		ax.set_ylabel('Beam MIN [pixels]')
		ax.set_zlabel('Beam PA [degrees]')
		plt.savefig(f'{test_dir}/beam_3d.png', dpi=500)
		plt.close(fig)

		fig, ax = plt.subplots()
		ax.scatter(X[:, 0], X[:, 1], c=self.kmeans.labels_.astype(float))
		ax.set_xlabel('Beam MAJ [pixels]')
		ax.set_ylabel('Beam MIN [pixels]')
		plt.savefig(f'{test_dir}/beam_maj_min.png', dpi=500)
		plt.close(fig)

		fig, ax = plt.subplots()
		ax.scatter(X[:, 1], X[:, 2], c=self.kmeans.labels_.astype(float))
		ax.set_xlabel('Beam MIN [pixels]')
		ax.set_ylabel('Beam PA [pixels]')
		plt.savefig(f'{test_dir}/beam_min_pa.png', dpi=500)
		plt.close(fig)

		fig, ax = plt.subplots()
		ax.scatter(X[:, 2], X[:, 0], c=self.kmeans.labels_.astype(float))
		ax.set_xlabel('Beam PA [pixels]')
		ax.set_ylabel('Beam MAJ [pixels]')
		plt.savefig(f'{test_dir}/beam_pa_min.png', dpi=500)
		plt.close(fig)
	
	def draw_b_min_dist(self) -> None:
		if self.b_min is None:
			self._preprocess()
		X = self.b_min
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_title('Beam MIN distribution')
		ax.hist(X[np.where(X < 500)], bins=1000)
		ax.set_xlabel('B_MIN [pixels]')
		ax.set_ylabel('Number')
		ax.set_xlim(0, 50)
		plt.savefig(f'{self.test_dir}/beam_min_dist.png', dpi=500)
		plt.close(fig)
	
	def draw_b_maj_dist(self) -> None:
		if self.b_maj is None:
			self._preprocess()
		X = self.b_maj
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_title('Beam MAJ distribution')
		ax.hist(X[np.where(X < 500)], bins=1000)
		ax.set_xlabel('B_MAJ [pixels]')
		ax.set_ylabel('Number')
		ax.set_xlim(0, 100)
		plt.savefig(f'{self.test_dir}/beam_maj_dist.png', dpi=500)
		plt.close(fig)
	
	def draw_b_pa_dist(self) -> None:
		X = self.b_pa
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(1, 1, 1)
		ax.set_title('Beam PA distribution')
		ax.hist(X, bins=100)
		ax.set_xlabel('B_PA [degrees]')
		ax.set_ylabel('Number')
		plt.savefig(f'{self.test_dir}/beam_pa_dist.png', dpi=500)
		plt.close(fig)