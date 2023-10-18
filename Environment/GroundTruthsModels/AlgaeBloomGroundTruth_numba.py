import numpy as np
import numba
from scipy.ndimage import gaussian_filter
import time
import colorcet as cc


@numba.jit(nopython=True, cache=True)
def set_seed(value):
    np.random.seed(value)

@numba.jit(nopython=True)
def gradient_image(image):
    height, width = image.shape
    gradient_dir = np.empty_like(image, dtype=np.float64)
    gradient_mag = np.empty_like(image, dtype=np.float64)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.int32)
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.int32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx = 0
            gy = 0
            for ky in range(3):
                for kx in range(3):
                    gx += image[y + ky - 1, x + kx - 1] * sobel_x[ky, kx]
                    gy += image[y + ky - 1, x + kx - 1] * sobel_y[ky, kx]
            gradient_dir[y, x] = np.arctan2(gy, gx)
            gradient_mag[y, x] = np.sqrt(gx**2 + gy**2)

    return gradient_dir, gradient_mag

@numba.jit(nopython=True)
def gaussian_filter_2d(image, sigma):
    kernel_size = 5  # Asegurar un tamaño impar para el kernel
    radius = 2
    kernel = np.empty((kernel_size, kernel_size), dtype=np.float64)
    filtered_image = np.empty_like(image, dtype=np.float64)

    # Generar el kernel gaussiano
    sum_val = 0.0
    for y in range(kernel_size):
        for x in range(kernel_size):
            dx = x - radius
            dy = y - radius
            kernel[y, x] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            sum_val += kernel[y, x]

    # Normalizar el kernel
    kernel /= sum_val

    # Aplicar el filtro con evaluación en los bordes
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            acc = 0.0
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    px = x + kx - radius
                    py = y + ky - radius
                    if px >= 0 and py >= 0 and px < image.shape[1] and py < image.shape[0]:
                        acc += image[py, px] * kernel[ky, kx]
            filtered_image[y, x] = acc

    return filtered_image

@numba.jit(nopython=True)
def current_field(position, M, N):


        u = -(position[1] - M / 2) / np.linalg.norm(position - np.array([M,N])/2 + 1e-6) + np.random.rand()
        v = (position[0] - N / 2) / np.linalg.norm(position - np.array([M,N])/2 + 1e-6) + np.random.rand()
        
        
        return np.array((u*np.random.rand(), v*np.random.rand()))

@numba.jit(nopython=True)
def compute_algae_bloom_map(T, M, N, nav_map, dt=0.1):

    # Initialize the map
    algae_map = np.zeros((T, M, N))
    algae_smooth_map = np.zeros((T, M, N))

    gradient_direction_field = np.zeros((M, N))
    gradient_magnitude_field = np.zeros((M, N))

    # Compute the gradient direction field
    gradient = gradient_image(nav_map)
    gradient_direction_field = gradient[0]
    gradient_magnitude_field = gradient[1]

    number_of_starting_points = np.random.randint(1, 6)
    visitable_positions = np.argwhere(nav_map == 1)
    starting_points = np.zeros((number_of_starting_points, 2))

    for idx in range(number_of_starting_points):
        starting_points[idx] = visitable_positions[np.random.randint(0, visitable_positions.shape[0])]


    num_of_particles_per_bloom = np.array([np.random.randint(1,20) + 80 for _ in range(number_of_starting_points)])
    total_num_of_particles = num_of_particles_per_bloom.sum()

    particles = np.zeros((total_num_of_particles, 2))

    # Initialize the particles with a gaussian distribution around the starting points
    for idx, num in enumerate(num_of_particles_per_bloom):

        p0 = 0 if idx == 0 else num_of_particles_per_bloom[:idx].sum()
        p1 = num_of_particles_per_bloom[:idx+1].sum()
        particles[p0:p1, 0] = np.array([np.random.standard_normal() * 3+2*np.random.rand() + starting_points[idx][0] for _ in range(num)])
        particles[p0:p1, 1] = np.array([np.random.standard_normal() * 3+2*np.random.rand() + starting_points[idx][1] for _ in range(num)])
    
    active_particles = np.ones((particles.shape[0],), dtype=np.bool_)
    
    # Initialize the currents
    contour_currents_x = np.zeros((M, N))

    for t in range(T):
        # Compute the currents
        for idx in range(total_num_of_particles):

            boundary_magnitude = gradient_magnitude_field[int(particles[idx][0]), int(particles[idx][1])]
            boundary_direction = gradient_direction_field[int(particles[idx][0]), int(particles[idx][1])]
            contour_current_x = boundary_magnitude * np.cos(boundary_direction)
            contour_current_y = boundary_magnitude * np.sin(boundary_direction)
            contour_current = np.array((contour_current_y, contour_current_x)) * 10
            particles[idx] = particles[idx] + (contour_current + current_field(particles[idx], M, N)) * dt
        

        # Update map
        for idx, particle in enumerate(particles):
            x = int(particle[0])
            y = int(particle[1])
            if x >= 0 and x < M and y >= 0 and y < N and nav_map[x,y] == 1 and active_particles[idx]:
                algae_map[t, x, y] = 1.0
            else:
                active_particles[idx] = False

        # Apply gaussian filter
        algae_smooth_map[t] = gaussian_filter_2d(algae_map[t], 0.8) 

    return algae_smooth_map * nav_map

class AlgaeBloomGroundTruthNumba:

    def __init__(self, grid, dt, max_steps = 120, t0 = 0):

        self.grid = grid
        self.dt = dt
        self.t = t0
        self.t0 = t0
        self.algae_bloom_map = None
        self.max_steps = max_steps
        self.fig = None

    def reset(self):

        self.algae_bloom_map = compute_algae_bloom_map(self.max_steps, self.grid.shape[0], self.grid.shape[1], self.grid, dt=self.dt)
        self.t = self.t0

    def step(self):

        self.t += 1
        assert self.t < self.algae_bloom_map.shape[0], "Simulation has ended. Increase max_steps."

    def read(self, position = None):
        
        if position is None:
            return self.algae_bloom_map[self.t,:,:]
        else:
            return self.algae_bloom_map[self.t, position[:,0].astype(int), position[:,1].astype(int)]

    def render(self):
        
        f_map = self.algae_bloom_map[self.t]
        f_map[self.grid == 0] = np.nan

        if self.fig is None:
            self.fig, self.ax = plt.subplots(1,1)
            self.d = self.ax.imshow(f_map, cmap = cc.cm['bgyw'], vmin=0.0, vmax = 1.0, interpolation='nearest')
            
            background = self.grid.copy()
            background[background == 1] = np.nan
            self.ax.imshow(background, cmap='gray')
            
        else:
            self.d.set_data(f_map)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    plt.ion()

    T = 100
    M = 58
    N = 60
    nav_map = np.genfromtxt("Environment\Maps\map.txt")

    set_seed(10)
    
    gt = AlgaeBloomGroundTruthNumba(nav_map, dt=0.1)

    gt.reset()
    gt.render()

    for _ in range(50):
        t0 = time.time()
        gt.reset()
        #gt.render()

        for t in range(100):
            gt.step()
            gt.render()
        
        print(time.time() - t0)
        










    



