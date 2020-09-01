import unittest
from tfrt import *
import numpy.testing as npt


class MyTestCase(unittest.TestCase):
    def test_pyramidsarray(self):
        pyramidss = PyramidArray(tf.constant([0., 0., 0.]), 1, 0.5, (4, 4), reflectivity=0.1)

        rays = Ray(p0=tf.constant([[0.2, 0.4, 2.], [0.2, 0.4, -2.], [2., 1.5, 0.5]], dtype=precision),
                   p1=tf.constant([[0., 0., -1.], [0., 0., 1.], [-1., 0., -1.]], dtype=precision),
                   intensity=tf.ones(3),
                   interact_num=tf.zeros(3, dtype=tf.int32))

        pt = pyramidss.intersect(rays)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0.2, 0.4, 0.2], [0.2, 0.4, 0.], [1.75, 1.5, 0.25]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[-1., 0., 0.], [0., 0., -1.], [1., 0., 1.]]))
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.2, 0.4, 0.2], [0.2, 0.4, 0.], [1.75, 1.5, 0.25]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.], [1., 0., 1.]]))
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.2, 0.4, 0.2], [0.2, 0.4, 0.], [1.75, 1.5, 0.25]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.], [1., 0., 1.]]))
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.2, 0.4, 0.2], [0.2, 0.4, 0.], [1.75, 1.5, 0.25]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.], [1., 0., 1.]]))

        npt.assert_array_almost_equal(pt.interact_num.numpy(), np.array([2, 1, 1]))

    def test_pyramidsspacing(self):
        pyramidss = PyramidArray(tf.constant([0., 0., 0.]), 1, 0.5, (4, 4), spacing=1., reflectivity=0.1)

        rays = Ray(p0=tf.constant([[0.2, 0.4, 2.]], dtype=precision),
                   p1=tf.constant([[0., 0., -1.]], dtype=precision),
                   intensity=tf.ones(1),
                   interact_num=tf.zeros(1, dtype=tf.int32))

        pt = pyramidss.intersect(rays)
        pt = pyramidss.intersect(pt)
        pt = pyramidss.intersect(pt)
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0.2, 0.4, 0.]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.]]))

        npt.assert_array_almost_equal(pt.interact_num.numpy(), np.array([1]))

        pyramidss = PyramidArray(tf.constant([0., 0., 0.]), 1, 0.5, (4, 4), spacing=0.1, reflectivity=0.1)

        rays = Ray(p0=tf.constant([[0.2, 0.4, 2.]], dtype=precision),
                   p1=tf.constant([[0., 0., -1.]], dtype=precision),
                   intensity=tf.ones(1),
                   interact_num=tf.zeros(1, dtype=tf.int32))

        pt = pyramidss.intersect(rays)
        pt = pyramidss.intersect(pt)
        pt = pyramidss.intersect(pt)
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.2, 0.4, 0.15]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.]]))

        npt.assert_array_almost_equal(pt.interact_num.numpy(), np.array([2]))

    def test_cone(self):
        cone = Cone(tf.constant([0., 0., 0.]), 1., 1., reflectivity=1)

        # test with single ray
        rays = Ray(p0=tf.constant([[0., -2., 0.5]], dtype=precision),
                   p1=tf.constant([[0., 1., 0.]], dtype=precision),
                   intensity=tf.ones(1),
                   interact_num=tf.zeros(1, dtype=tf.int32))

        pt = cone.intersect(rays)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0., -0.5, 0.5]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.]]))

        # test with multiple rays
        rays = Ray(p0=tf.constant([[0., -2., 0.5], [0., -2., -0.5], [0.5, 0., 2.], [0.5, 0., 0.7]], dtype=precision),
                   p1=tf.constant([[0., 1., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., -1.]], dtype=precision),
                   intensity=tf.ones(4),
                   interact_num=tf.zeros(4, dtype=tf.int32))

        pt = cone.intersect(rays)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0., -0.5, 0.5], [0., -2., -0.5], [0.5, 0., 0.5],
                                                               [0.5, 0., 0.5]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.],
                                                               [1., 0., 0.]]))

        # test with multiple rays from behind
        rays = Ray(p0=tf.constant([[0., 2., 0.5], [0., 2., 1.5]], dtype=precision),
                   p1=tf.constant([[0., -1., 0.], [0., 1., 0.]], dtype=precision),
                   intensity=tf.ones(2),
                   interact_num=tf.zeros(2, dtype=tf.int32))

        pt = cone.intersect(rays)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0., 0.5, 0.5], [0., 2., 1.5]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 1., 0.]]))

    def test_conesarray(self):
        coness = ConeArray(tf.constant([0., 0., 0.]), 1, 1., (2, 2), reflectivity=1.)

        rays = Ray(p0=tf.constant([[0.1, 1., 2.], [0.2, 0.4, -2.], [0.1, 1., .2]], dtype=precision),
                   p1=tf.constant([[0., 0., -1.], [0., 0., 1.], [0., 0., -1.]], dtype=precision),
                   intensity=tf.ones(3),
                   interact_num=tf.zeros(3, dtype=tf.int32))

        pt = coness.intersect(rays)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0.1, 1., 0.1], [0.2, 0.4, 0.], [0.1, 1., 0.1]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[-1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]]))

        pt = coness.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.1, 1., 0.1], [0.2, 0.4, 0.], [-0.1, 1., 0.1]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.], [0., 0., 1.]]))
        pt = coness.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.1, 1., 0.1], [0.2, 0.4, 0.], [-0.1, 1., 0.1]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.], [0., 0., 1.]]))
        pt = coness.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.1, 1., 0.1], [0.2, 0.4, 0.], [-0.1, 1., 0.1]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.], [0., 0., 1.]]))

        npt.assert_array_almost_equal(pt.interact_num.numpy(), np.array([2, 1, 2]))

    def test_conesdensearray(self):
        coness = ConeDenseArray(center=tf.constant([0., 0., 0.]),
                                radius=1.,
                                coneheight=1.,
                                width=4,
                                height=4,
                                reflectivity=0.1)

        rays = Ray(p0=tf.constant([[0.1, 1., 2.], [0.2, 0.4, -2.], [0.1, 1., .2]], dtype=precision),
                   p1=tf.constant([[0., 0., -1.], [0., 0., 1.], [0., 0., -1.]], dtype=precision),
                   intensity=tf.ones(3),
                   interact_num=tf.zeros(3, dtype=tf.int32))

        print(coness.x, coness.y, coness.top_left, coness.top_right, coness.bottom_left, coness.bottom_right)

        pt = coness.intersect(rays)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0.1, 1., 0.1], [0.2, 0.4, 0.], [0.1, 1., 0.1]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[-1., 0., 0.], [0., 0., -1.], [-1., 0., 0.]]))


if __name__ == '__main__':
    unittest.main()
