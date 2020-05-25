# Taichi crash on Colab workaround (see https://github.com/taichi-dev/taichi/issues/235)
# Part 1
import os, json, signal, time
if 'libtcmalloc' in os.environ.get('LD_PRELOAD', ''):
  kernel_fn = '/usr/local/share/jupyter/kernels/python3/kernel.json'
  spec = json.load(open(kernel_fn))
  spec['env'] = {'LD_PRELOAD': ''}
  json.dump(spec, open(kernel_fn, 'w'))
  print("Installed Taichi workaround. Don't wait for this cell to finish,")
  print('just REFRESH the browser tab and RUN this cell again.', flush=True)
  time.sleep(0.5)
  # killing the kernel manager so that specs get reloaded
  os.kill(os.getppid(), signal.SIGTERM)
else:
  print('Kernel is Taichi-ready!')

# Part 2
# !pip install taichi-nightly-cuda-10-1==0.5.2

#Part 3
# from 'mpm88.py'

import taichi as ti
import random

ti.reset()

ti.init(arch=ti.cuda)

dim = 2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2.0e-4
p_vol = (dx * 0.5)**2
p_rho = 1
p_mass = p_vol * p_rho
E = 400
img_size = 512

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)
J = ti.var(dt=ti.f32, shape=n_particles)
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))
grid_img = ti.Vector(3, dt=ti.f32, shape=(img_size, img_size))

@ti.kernel
def substep():
  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)]
    stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
    affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        offset = ti.Vector([i, j])
        dpos = (offset.cast(float) - fx) * dx
        weight = w[i][0] * w[j][1]
        grid_v[base + offset].atomic_add(
            weight * (p_mass * v[p] + affine @ dpos))
        grid_m[base + offset].atomic_add(weight * p_mass)

  for i, j in grid_m:
    if grid_m[i, j] > 0:
      bound = 3
      inv_m = 1 / grid_m[i, j]
      grid_v[i, j] = inv_m * grid_v[i, j]
      grid_v[i, j][1] -= dt * 9.8
      if i < bound and grid_v[i, j][0] < 0:
        grid_v[i, j][0] = 0
      if i > n_grid - bound and grid_v[i, j][0] > 0:
        grid_v[i, j][0] = 0
      if j < bound and grid_v[i, j][1] < 0:
        grid_v[i, j][1] = 0
      if j > n_grid - bound and grid_v[i, j][1] > 0:
        grid_v[i, j][1] = 0

  for p in x:
    base = (x[p] * inv_dx - 0.5).cast(int)
    fx = x[p] * inv_dx - base.cast(float)
    w = [
        0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0), 0.5 * ti.sqr(fx - 0.5)
    ]
    new_v = ti.Vector.zero(ti.f32, 2)
    new_C = ti.Matrix.zero(ti.f32, 2, 2)
    for i in ti.static(range(3)):
      for j in ti.static(range(3)):
        dpos = ti.Vector([i, j]).cast(float) - fx
        g_v = grid_v[base + ti.Vector([i, j])]
        weight = w[i][0] * w[j][1]
        new_v += weight * g_v
        new_C += 4 * weight * ti.outer_product(g_v, dpos) * inv_dx
    v[p] = new_v
    x[p] += dt * v[p]
    J[p] *= 1 + dt * new_C.trace()
    C[p] = new_C

    # draw
    xi = (x[p] * img_size).cast(int)
    grid_img[xi] += ti.Vector([1.0, 1.0, 1.0])

#part 4
import PIL.Image, PIL.ImageFont, PIL.ImageDraw

def gen_points(s, font_size=42):
  font = PIL.ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf', font_size)
  w, h = font.getsize(s)
  im = PIL.Image.new('L', (w, h))
  draw  = PIL.ImageDraw.Draw(im)
  draw.text((0, 0), s, fill=255, font=font)
  im = np.uint8(im)
  y, x = np.float32(im.nonzero())
  pos = np.column_stack([x, y]) / w
  return pos

pos = np.vstack([
  gen_points(' Taichi ', 100)+(0.0, 0.1),
  gen_points('   +   ', 100)+(0.0, 0.3),
  gen_points(' Colab ', 100)+(0.0, 0.5),
])
pos[:,1] = 1.0-pos[:,1]
np.random.shuffle(pos)
pos = pos[:n_particles]

for i in range(n_particles):
  x[i] = pos[i]
  v[i] = [0, 0]
  J[i] = 1



with VideoWriter('mpm88.mp4', 60) as vid:
  frames = []
  for frame in tqdm.trange(60*5):
    grid_img.fill([0, 0, 0])
    for s in range(20):
      grid_v.fill([0, 0])
      grid_m.fill(0)
      substep()
    vis = grid_img.to_numpy()[...,0]/20
    vis **= 1/2.2  # gamma correction
    vis = vis.swapaxes(0, 1)[::-1]
    frames.append(vis)
    vid.add(vis)

  for vis in tqdm.tqdm(frames[::-1]):
    vid.add(vis)


#Part 5
mvp.ipython_display('mpm88.mp4', loop=True)