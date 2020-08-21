import unittest
from tfrt import *
import numpy.testing as npt


class MyTestCase(unittest.TestCase):
    def test_FirstTest(self):
        pyramidss = PyramidArray(tf.constant([0., 0., 0.]), 1, 0.5, (4, 4), reflectivity=0.1)

        rays = Ray(p0=tf.constant([[0.2, 0.4, 2.], [0.2, 0.4, -2.]], dtype=precision),
                   p1=tf.constant([[0., 0., -1.], [0., 0., 1.]], dtype=precision),
                   intensity=tf.ones(2),
                   interact_num=tf.zeros(2, dtype=tf.int32))

        pt = pyramidss.intersect(rays)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[0.2, 0.4, 0.2], [0.2, 0.4, 0.]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[-1., 0., 0.], [0., 0., -1.]]))
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.2, 0.4, 0.2], [0.2, 0.4, 0.]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.]]))
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.2, 0.4, 0.2], [0.2, 0.4, 0.]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.]]))
        pt = pyramidss.intersect(pt)
        npt.assert_array_almost_equal(pt.p0.numpy(), np.array([[-0.2, 0.4, 0.2], [0.2, 0.4, 0.]]))
        npt.assert_array_almost_equal(pt.p1.numpy(), np.array([[0., 0., 1.], [0., 0., -1.]]))

        npt.assert_array_almost_equal(pt.interact_num.numpy(), np.array([2, 1]))


if __name__ == '__main__':
    unittest.main()
