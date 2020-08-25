from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import tensorflow as tf

# source: http://geomalgorithms.com/a06-_intersect-2.html
# source: https://www.erikrotteveel.com/python/three-dimensional-ray-tracing-in-python/

faraway = 99999  # faraway distance
precision = tf.float64  # default precision
pi = tf.constant(np.pi, dtype=precision)

# for numerical stability epsilon
if precision == tf.float32:
    epsilon = tf.constant(1.e-07, precision)
elif precision == tf.float64:
    epsilon = tf.constant(1.e-15, precision)


def set_precision(p):
    global precision
    precision = p


def mag(tensor):
    """
    Calculate magnitude of the vector, return scalar tensor
    """
    if tf.equal(tensor.get_shape().rank, 1):
        mag = tf.sqrt(tf.tensordot(tensor, tensor, 1))
    else:
        mag = tf.sqrt(tf.reduce_sum(tensor*tensor, 1))
    return mag


def ray_reflection(rays, normal):
    """
    Calculate reflection of rays `rays` with normal `normal`
    `
    :param rays: Rays directional vector, shape Nx3
    :type rays: Ray
    :param normal: normal vector
    :type normal: tf.Tensor
    """
    ray_direction = rays.p1 - tf.multiply(normal, tf.expand_dims(tf.reduce_sum(normal * rays.p1, 1), 1)) * 2.
    # if directional vector small enough, then assume 0.
    ray_direction = tf.where(tf.greater(tf.abs(ray_direction), epsilon), ray_direction, tf.zeros_like(ray_direction))

    return ray_direction


def norm(tensor):
    """
    Calculate norm of the vector, return normalized vector
    """
    _mag = mag(tensor)
    if tf.equal(tensor.get_shape().rank, 1):
        return tensor * (1.0 / tf.where(tf.less_equal(_mag, epsilon), tf.ones_like(_mag), _mag))
    else:
        return tensor * tf.expand_dims(1.0 / tf.where(tf.less_equal(_mag, epsilon), tf.ones_like(_mag), _mag), 1)

    
def tile_vector(tensor, num):
    return tf.tile(tf.expand_dims(tensor, 0), [num, 1])


def polar(tensor):
    _norm = norm(tensor)
    phi, theta = tf.math.atan2((_norm[:, 0]+epsilon), _norm[:, 1]), tf.math.acos(_norm[:, 2])

    return tf.where(tf.less(phi, 0.), 2*pi+phi, phi), theta


class Ray:
    def __init__(self, p0, p1, intensity, interact_num):
        """
        Basic Ray class, originating from `p0` with a directional vector of `p1
        `
        :param p0: 3D vectors for the origins of rays
        :type p0: tf.Tensor
        :param p1: 3D vectors for the origins of rays
        :type p1: tf.Tensor
        :param intensity: Initial intensity of rays
        :type intensity: tf.Tensor
        :param interact_num: Initial number of interaction experienced by rays
        :type interact_num: tf.Tensor
        """
        self.p0 = p0  # ray origins
        self.p1 = p1  # ray direction
        self.intensity = intensity
        self.interact_num = interact_num

        p0_rows, p0_columns = p0.get_shape()

        tf.debugging.assert_equal(tf.size(self.p0), tf.size(self.p1), message="Rays shape not equal")
        tf.debugging.assert_equal(p0_rows, tf.size(self.intensity), message="Rays shape not equal")
        tf.debugging.assert_equal(p0_rows, tf.size(self.interact_num), message="Rays shape not equal")

    def __getitem__(self, key):
        return Ray(self.p0[key], self.p1[key], self.intensity[key], self.interact_num[key])

    def __setitem__(self, key, value):
        if key.dtype == tf.bool:
            key_3 = tf.concat([tf.expand_dims(key, 1), tf.expand_dims(key, 1), tf.expand_dims(key, 1)], 1)
            self.p0 = tf.where(key_3, value.p0, self.p0)
            self.p1 = tf.where(key_3, value.p1, self.p1)
            self.intensity = tf.where(key, value.intensity, self.intensity)
            self.interact_num = tf.where(key, value.interact_num, self.interact_num)
        else:
            self.p0[key] = value.p0
            self.p1[key] = value.p1
            self.intensity[key] = value.intensity
            self.interact_num[key] = value.interact_num

    def size(self):
        num_rays = tf.size(self.p0) // 3
        return num_rays

    def copy(self):
        return deepcopy(self)


class Surface(ABC):
    """
    Basic class for surfaces
    """
    def __init__(self):
        pass

    @abstractmethod
    def vertices(self):
        pass


class Triangle(Surface):
    def __init__(self, v0, v1, v2, reflectivity=1.):
        """
        A triangle with vertices `v0`, `v1`, `v2` and `reflectivity`

        :param v0: 3D vectors for a vertex
        :type v0: tf.Tensor
        :param v1: 3D vectors for a vertex
        :type v1: tf.Tensor
        :param v2: 3D vectors for a vertex
        :type v2: tf.Tensor
        :param reflectivity: Reflectivity of the surface
        :type reflectivity: float
        """
        super().__init__()
        self.v0 = tf.cast(v0, precision)
        self.v1 = tf.cast(v1, precision)
        self.v2 = tf.cast(v2, precision)

        self.u = self.v1 - self.v0
        self.v = self.v2 - self.v0
        self.reflectivity = reflectivity
        
        self.normal = norm(tf.linalg.cross(self.u, self.v))

    @property
    def vertices(self):
        return tf.stack([self.v0, self.v1, self.v2])

    def intersect(self, rays):
        num_rays = rays.size()
        
        tiled_v = tile_vector(self.v, num_rays)
        tiled_u = tile_vector(self.u, num_rays)
        tiled_normal = tile_vector(self.normal, num_rays)
                
        b = tf.reduce_sum(tiled_normal*rays.p1, 1)
        a = tf.reduce_sum(tiled_normal*(self.v0 - rays.p0), 1)
        
        # check if the ray is close enough to be parallel or close enough to lie in the plane
        cond_0_1 = tf.greater(tf.abs(b), epsilon)
        cond_0_2 = tf.greater(tf.abs(a), epsilon)
        cond_0 = tf.logical_and(cond_0_1, cond_0_2)

        rI = tf.expand_dims(tf.where(tf.logical_or(cond_0, tf.less(a/b, 0.)), a/b, tf.zeros_like(a)), -1)
        
        p_intersect = rays.p0 + rays.p1 * rI

        w = p_intersect - self.v0  # p0 + rI * p1 - v0
        
        wv_dot = tf.reduce_sum(w*tiled_v, 1)
        wu_dot = tf.reduce_sum(w*tiled_u, 1)
        
        uv_dot = tf.tensordot(self.u, self.v, 1)
        uu_dot = tf.tensordot(self.u, self.u, 1)
        vv_dot = tf.tensordot(self.v, self.v, 1)
        
        denom = uv_dot * uv_dot - uu_dot * vv_dot
        si = (uv_dot * wv_dot - vv_dot * wu_dot) / denom
        ti = (uv_dot * wu_dot - uu_dot * wv_dot) / denom
        
        ray_direction = ray_reflection(rays, tiled_normal)
        
        cond_1 = tf.less(tf.squeeze(rI), epsilon)
        cond_2 = tf.less(si, 0.)
        cond_3 = tf.greater(si, 1.)
        cond_4 = tf.less(ti, 0.)
        cond_5 = tf.greater(si + ti, 1.)
        
        no_interaction_idx = tf.logical_or(tf.logical_or(tf.logical_or(tf.logical_or(cond_1, cond_2), cond_3), cond_4), cond_5)
        no_interaction_idx_3 = tf.concat([tf.expand_dims(no_interaction_idx, 1), tf.expand_dims(no_interaction_idx, 1), tf.expand_dims(no_interaction_idx, 1)], 1)

        _p_intersect = tf.where(no_interaction_idx_3, rays.p0, p_intersect)
        ray_direction = tf.where(no_interaction_idx_3, rays.p1, ray_direction)
        new_interact_num = tf.where(no_interaction_idx, rays.interact_num, rays.interact_num+1)
        new_intensity = tf.where(no_interaction_idx, rays.intensity, rays.intensity*self.reflectivity)
        
        return Ray(_p_intersect, ray_direction, intensity=new_intensity, interact_num=new_interact_num)

    
class Plane(Surface):
    def __init__(self, v0, v1, v2, v3, reflectivity=1.):
        """
        A plane with vertices `v0`, `v1`, `v2` and `reflectivity`

        :param v0: 3D vectors for a vertex
        :type v0: tf.Tensor
        :param v1: 3D vectors for a vertex
        :type v1: tf.Tensor
        :param v2: 3D vectors for a vertex
        :type v2: tf.Tensor
        :param v3: 3D vectors for a vertex
        :type v3: tf.Tensor
        :param reflectivity: Reflectivity of the surface
        :type reflectivity: tf.Tensor
        """
        super().__init__()
        self.v0 = tf.cast(v0, precision)
        self.v1 = tf.cast(v1, precision)
        self.v2 = tf.cast(v2, precision)
        self.v3 = tf.cast(v3, precision)

        self.u = self.v1 - self.v0
        self.v = self.v3 - self.v0
        self.reflectivity = reflectivity
        
        self.normal = norm(tf.linalg.cross(self.u, self.v))

    @property
    def vertices(self):
        return tf.stack([self.v0, self.v1, self.v2, self.v3])

    def intersect(self, rays):
        num_rays = rays.size()
        
        tiled_v = tile_vector(self.v, num_rays)
        tiled_u = tile_vector(self.u, num_rays)
        tiled_normal = tile_vector(self.normal, num_rays)
        
        b = tf.reduce_sum(tiled_normal*rays.p1, 1)
        a = tf.reduce_sum(tiled_normal*(self.v0 - rays.p0), 1)
        # check if the ray is close enough to be parallel or close enough to lie in the plane
        cond_0_1 = tf.greater(tf.abs(b), epsilon)
        cond_0_2 = tf.greater(tf.abs(a), epsilon)
        cond_0 = tf.logical_and(cond_0_1, cond_0_2)

        rI = tf.expand_dims(tf.where(tf.logical_or(cond_0, tf.less(a/b, 0.)), a/b, tf.zeros_like(a)), -1)
        
        p_intersect = rays.p0 + rays.p1 * rI

        w = p_intersect - self.v0  # p0 + rI * p1 - v0
        
        wv_dot = tf.reduce_sum(w*tiled_v, 1)
        wu_dot = tf.reduce_sum(w*tiled_u, 1)
        
        uv_dot = tf.tensordot(self.u, self.v, 1)
        uu_dot = tf.tensordot(self.u, self.u, 1)
        vv_dot = tf.tensordot(self.v, self.v, 1)
        
        denom = uv_dot * uv_dot - uu_dot * vv_dot
        si = (uv_dot * wv_dot - vv_dot * wu_dot) / denom
        ti = (uv_dot * wu_dot - uu_dot * wv_dot) / denom
        
        ray_direction = ray_reflection(rays, tiled_normal)
        
        cond_1 = tf.less(tf.squeeze(rI), epsilon)
        cond_2 = tf.less(si, 0.)
        cond_3 = tf.greater(si, 1.)
        cond_4 = tf.less(ti, 0.)
        cond_5 = tf.greater(ti, 1.)
        
        no_interaction_idx = tf.logical_or(tf.logical_or(tf.logical_or(tf.logical_or(cond_1, cond_2), cond_3), cond_4), cond_5)
        no_interaction_idx_3 = tf.concat([tf.expand_dims(no_interaction_idx, 1), tf.expand_dims(no_interaction_idx, 1), tf.expand_dims(no_interaction_idx, 1)], 1)
        
        p_intersect = tf.where(no_interaction_idx_3, rays.p0, p_intersect)
        ray_direction = tf.where(no_interaction_idx_3, rays.p1, ray_direction)
        new_interact_num = tf.where(no_interaction_idx, rays.interact_num, rays.interact_num+1)
        new_intensity = tf.where(no_interaction_idx, rays.intensity, rays.intensity*self.reflectivity)
        
        return Ray(p_intersect, ray_direction, intensity=new_intensity, interact_num=new_interact_num)    


class Pyramid:
    def __init__(self, center, width, height, reflectivity=1.):
        """
        A pyramid

        :param center: 3D vectors for the center of the base
        :type center: tf.Tensor
        :param width: width of the base
        :type width: float
        :param height: height of the base
        :type height: float
        :param reflectivity: Reflectivity of the surface
        :type reflectivity: float
        """
        self.center = tf.cast(center, precision)  # center of the pyramid base
        self.width = tf.cast(width, precision)  # width of the pyramid base
        self.height = tf.cast(height, precision)
        self.reflectivity = reflectivity
        
        self.top_left = self.center + tf.stack([-1. * self.width / 2., self.width / 2., 0.])
        self.top_right = self.center + tf.stack([self.width / 2., self.width / 2., 0.])
        self.bottom_left = self.center + tf.stack([-1. * self.width / 2., -1. * self.width / 2., 0.])
        self.bottom_right = self.center + tf.stack([self.width / 2., -1. * self.width / 2., 0.])
        self.top_v = self.center + tf.stack([0., 0., self.height])
        
        self.vertices = tf.stack([self.top_left, self.top_right, self.bottom_right, self.bottom_left, self.top_v])
        
        self.tri_1 = Triangle(self.top_v, self.top_left, self.top_right, self.reflectivity)
        self.tri_2 = Triangle(self.top_v, self.top_right, self.bottom_right, self.reflectivity)
        self.tri_3 = Triangle(self.top_v, self.bottom_right, self.bottom_left, self.reflectivity)
        self.tri_4 = Triangle(self.top_v, self.bottom_left, self.top_left, self.reflectivity)
        self.tris = [self.tri_1, self.tri_2, self.tri_3, self.tri_4]
        
    def intersect(self, rays):
        _pt = deepcopy(rays)  # by default assume not intersecting with pyramid
        distance = tf.ones(rays.size(), dtype=precision) * faraway
        
        for tri in self.tris:
            pt = tri.intersect(rays)
            interacted_idx = tf.greater(pt.interact_num, rays.interact_num)
            dist = mag(rays.p0-pt.p0)  # get the distance
            interacted_w_shortest_idx = tf.logical_and(interacted_idx, tf.less(dist, distance))

            if tf.math.count_nonzero(interacted_w_shortest_idx) == 0:
                continue
            else:
                distance = tf.where(interacted_w_shortest_idx, dist, distance)
                # its fine, weird indexing
                _pt[interacted_w_shortest_idx] = pt
        return _pt


class Cone:
    def __init__(self, center, radius, height, reflectivity=1.):
        """
        A Cone where the base centered at `center` with height `height`

        :param center: 3D vectors for the center of the base
        :type center: tf.Tensor
        :param radius: radius of the base
        :type radius: float
        :param height: height of the base
        :type height: float
        :param reflectivity: Reflectivity of the surface
        :type reflectivity: float
        """
        self.center = tf.cast(center, precision)
        self.radius = tf.cast(radius, precision)
        self.height = tf.cast(height, precision)
        self.reflectivity = reflectivity

        self.c = self.center + tf.constant([0., 0., height], dtype=precision)  # vector for the tips
        self.v = self.center - self.c  # vector for the axis
        self.halfangle = tf.atan(self.radius/self.height)
        self.halfangle2 = tf.cos(self.halfangle)**2

    def intersect(self, rays):
        # see http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone/
        num_rays = rays.size()

        tiled_v = tile_vector(self.v, num_rays)
        tiled_c = tile_vector(self.c, num_rays)

        co = rays.p0 - tiled_c
        p1v_dot = tf.reduce_sum(rays.p1 * tiled_v, 1)

        a = p1v_dot * tf.reduce_sum(rays.p1 * tiled_v, 1) - self.halfangle2
        b = 2. * (p1v_dot * tf.reduce_sum(tiled_v * co, 1) - tf.reduce_sum(co * rays.p1, 1) * self.halfangle2)
        c = tf.reduce_sum(co * tiled_v, 1) ** 2 - tf.reduce_sum(co * co, 1) * self.halfangle2

        det = b * b - 4. * a * c
        det = tf.where(tf.greater(det, 0.), tf.sqrt(det), tf.ones_like(det) * -1.)

        t1 = (-b - det) / (2. * a)
        t2 = (-b + det) / (2. * a)

        t = tf.where(tf.logical_and(tf.logical_or(tf.less(t1, 0.),  tf.greater(t2, 0.)),
                                    tf.less(t2, t1)), t2, t1)

        p_intersect = rays.p0 + tf.multiply(rays.p1, tf.expand_dims(t,1))
        cp = p_intersect - tiled_c

        h = tf.reduce_sum(cp * tiled_v, 1)

        normal = norm(tf.multiply(cp, tf.expand_dims(tf.reduce_sum(tiled_v * cp, 1) / tf.reduce_sum(cp * cp, 1), 1)) -
                      tiled_v)

        ray_direction = ray_reflection(rays, normal)

        cond_1 = tf.less(det, 0.)
        cond_2 = tf.less(t, 0.)
        cond_3 = tf.logical_or(tf.less(h, 0.), tf.greater(h, self.height))

        no_interaction_idx = tf.logical_or(tf.logical_or(cond_1, cond_2), cond_3)
        no_interaction_idx_3 = tf.concat([tf.expand_dims(no_interaction_idx, 1), tf.expand_dims(no_interaction_idx, 1),
                                          tf.expand_dims(no_interaction_idx, 1)], 1)

        p_intersect = tf.where(no_interaction_idx_3, rays.p0, p_intersect)
        ray_direction = tf.where(no_interaction_idx_3, rays.p1, ray_direction)
        new_interact_num = tf.where(no_interaction_idx, rays.interact_num, rays.interact_num + 1)
        new_intensity = tf.where(no_interaction_idx, rays.intensity, rays.intensity * self.reflectivity)

        return Ray(p_intersect, ray_direction, intensity=new_intensity, interact_num=new_interact_num)


class PyramidArray:
    def __init__(self, center, width, height, resolution, spacing=0., reflectivity=0.1):
        """
        An array of  pyramid

        :param center: 3D vectors for the center of the base
        :type center: tf.Tensor
        :param width: width of the base
        :type width: float
        :param height: height of the base
        :type height: float
        :param resolution: number of pyramid at each side
        :type resolution: tuple
        :param spacing: spacing between each pyramid
        :type spacing: float
        :param reflectivity: Reflectivity of the surface
        :type reflectivity: float
        """
        self.center = tf.cast(center, precision)  # detector center
        self.width = tf.cast(width, precision)  # pixel width
        self.height = tf.cast(height, precision)  # pixel width
        self.resolution = resolution  # resolution (W x H)
        self.spacing = spacing  # spacing between pyramids
        self.reflectivity = reflectivity
        
        self.num_pixel = tf.reduce_prod(self.resolution)        
        self.x_append = (self.resolution[0] - 1) * self.spacing  # total extra space from spacing
        self.y_append = (self.resolution[1] - 1) * self.spacing  # total extra space from spacing
        self.x, self.y = self.pixels_locations() # center of each pyramid

        self.top_left = self.center + tf.stack([-1. * self.width * self.resolution[0] / 2. - self.x_append/2, self.width * self.resolution[1] / 2. + self.y_append/2, 0.])
        self.top_right = self.center + tf.stack([self.width * self.resolution[0] / 2. + self.x_append/2, self.width * self.resolution[1] / 2. + self.x_append/2, 0.])
        self.bottom_left = self.center + tf.stack([-1. * self.width * self.resolution[0] / 2. - self.x_append/2, -1. * self.width * self.resolution[1] / 2. - self.x_append/2, 0.])
        self.bottom_right = self.center + tf.stack([self.width * self.resolution[0] / 2. + self.x_append/2, -1. * self.width * self.resolution[1] / 2. - self.y_append/2, 0.])
        
        self.pyramid_list = [self.get_pyramid_from_array(i) for i in range(self.num_pixel)]
        
        self.backplane = Plane(self.top_left, self.top_right, self.bottom_right, self.bottom_left)  # the plane where pyramids are sitting on, in case spacing != 0
        
    def pixels_locations(self):
        physical_w = self.width * self.resolution[0] + self.x_append
        physical_h = self.width * self.resolution[1] + self.y_append
        
        all_w = physical_w / 2. - (tf.linspace(tf.constant(0., dtype=precision), 
                                               tf.constant(self.resolution[0]-1., dtype=precision), 
                                               self.resolution[0]) * self.width) - self.width / 2.
        all_h = physical_h / 2. - (tf.linspace(tf.constant(0., dtype=precision), 
                                               tf.constant(self.resolution[1]-1., dtype=precision), 
                                               self.resolution[1]) * self.width) - self.width / 2.
        
        all_w = all_w - (np.array([range(0, self.resolution[0])]) * self.spacing)
        all_h = all_h - (np.array([range(0, self.resolution[0])]) * self.spacing)
        x, y = tf.meshgrid(all_w, all_h)
        
        return x, y

    def get_pyramid_from_array(self, i):
        assert i < self.num_pixel
        i = np.unravel_index(i, self.resolution)
        return Pyramid(self.center + tf.concat([self.x[i], self.y[i], 0.], 0), self.width, self.height, reflectivity=self.reflectivity)
    
    def intersect(self, rays):
        _pt = deepcopy(rays)  # by default assume not intersecting with pyramid
        distance = tf.ones(rays.size(), dtype=precision) * faraway
        
        for i in range(self.num_pixel):
            pt = self.pyramid_list[i].intersect(rays)
            interacted_idx = tf.greater(pt.interact_num, rays.interact_num)
            dist = mag(rays.p0-pt.p0)  # get the distance
            interacted_w_shortest_idx = tf.logical_and(interacted_idx, tf.less(dist, distance))
            if tf.math.count_nonzero(interacted_w_shortest_idx) == 0:
                continue
            else:
                distance = tf.where(interacted_w_shortest_idx, dist, distance)
                # its fine, weird indexing
                _pt[interacted_w_shortest_idx] = pt
        __pt = self.backplane.intersect(rays)
        interacted_idx = tf.greater(__pt.interact_num, rays.interact_num)
        dist = mag(rays.p0-__pt.p0)  # get the distance
        interacted_w_shortest_idx = tf.logical_and(interacted_idx, tf.less(dist, distance))
        _pt[interacted_w_shortest_idx] = __pt
        return _pt


class Detector:
    """
    A detector lying on xy plane
    """
    def __init__(self, center, resolution, pixel_width, reflectivity=0.):
        """
        A class for retengular detector

        :param center: 3D vectors for the center of the base
        :type center: tf.Tensor
        :param resolution: number of pyramid at each side
        :type resolution: tuple
        :param pixel_width: width of the pixel
        :type pixel_width: float
        :param reflectivity: Reflectivity of the surface
        :type reflectivity: float
        """
        self.center = tf.cast(center, precision)  # detector center
        self.pixel_width = tf.cast(pixel_width, precision)  # pixel width
        self.resolution = resolution  # resolution (W x H)
        
        self.num_pixel = tf.reduce_prod(self.resolution)
        self.x, self.y = self.pixels_locations()
        self.reflectivity = reflectivity
        
        self.top_left = self.center + tf.stack([-1. * self.pixel_width * self.resolution[0] / 2., self.pixel_width * self.resolution[1] / 2., 0.])
        self.top_right = self.center + tf.stack([self.pixel_width * self.resolution[0] / 2., self.pixel_width * self.resolution[1] / 2., 0.])
        self.bottom_left = self.center + tf.stack([-1. * self.pixel_width * self.resolution[0] / 2., -1. * self.pixel_width * self.resolution[1] / 2., 0.])
        self.bottom_right = self.center + tf.stack([self.pixel_width * self.resolution[0] / 2., -1. * self.pixel_width * self.resolution[1] / 2., 0.])
                
        self.u = self.top_right - self.top_left
        self.v = self.bottom_left - self.top_left
        
        self.normal = norm(tf.linalg.cross(self.u, self.v))
        
        self.plane = Plane(self.top_left, self.top_right, self.bottom_right, self.bottom_left)
        
    def pixels_locations(self):
        physical_w = self.pixel_width * self.resolution[0]
        physical_h = self.pixel_width * self.resolution[1]
        
        all_w = physical_w / 2. - (tf.linspace(tf.constant(0., dtype=precision), 
                                               tf.constant(self.resolution[0]-1., dtype=precision), 
                                               self.resolution[0]) * self.pixel_width) - self.pixel_width / 2.
        all_h = physical_h / 2. - (tf.linspace(tf.constant(0., dtype=precision), 
                                               tf.constant(self.resolution[1]-1., dtype=precision), 
                                               self.resolution[1]) * self.pixel_width) - self.pixel_width / 2.
        
        x, y = tf.meshgrid(all_w, all_h)
        
        return x, y
        
    def get_random_rays_from_pixel(self, i, num=1):
        assert i < self.num_pixel
        i = np.unravel_index(i, self.resolution)
        xi = tf.random.uniform([num, 1], self.x[i]-self.pixel_width/2, self.x[i]+self.pixel_width/2, precision)
        yi = tf.random.uniform([num, 1], self.y[i]-self.pixel_width/2, self.y[i]+self.pixel_width/2, precision)
        xdirecti = tf.random.uniform([num, 1], -2., 2., precision)
        ydirecti = tf.random.uniform([num, 1], -2., 2., precision)
        # hard code x-y direction minium to not waste ray??        
        return Ray(self.center + tf.concat([xi, yi, tf.zeros((num, 1), dtype=precision)], 1), 
                   tf.concat([xdirecti, ydirecti, tf.ones((num, 1), dtype=precision)*-1.], 1), 
                   intensity=tf.ones(num, dtype=precision), 
                   interact_num=tf.zeros(num, dtype=tf.int32))
    
    def intersect(self, rays):        
        return self.plane.intersect(rays)
