# Based on code from
# https://github.com/tody411/GuidedFilter

#  Implementation of guided filter.
#  * GuidedFilter: Original guided filter.
#  @author      tody
#  @date        2015/08/26

# The MIT License (MIT)
# 
# Copyright (c) 2015 tody
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# Modifications:
# - use float64
# - added support for pixel weights

import numpy as np
import cv2

## Return if the input image is gray or not.
def _isGray(I):
    return len(I.shape) == 2

def boxFilter(img, r):
	return cv2.blur(img, (r, r))
	#return cv2.GaussianBlur(img, (r, r), 0)


## Guide filter.
class GuidedFilter:
    ## Constructor.
    #  @param I Input guidance image. Color or gray.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.4, weights=None):
        I = np.array(I, dtype=np.float64)

        if _isGray(I):
            self._guided_filter = GuidedFilterGray(I, radius, epsilon, weights)
        else:
            self._guided_filter = GuidedFilterColor(I, radius, epsilon, weights)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._guided_filter.filter(p)


## Common parts of guided filter.
#
#  This class is used by guided_filter class. GuidedFilterGray and GuidedFilterColor.
#  Based on guided_filter._computeCoefficients, guided_filter._computeOutput,
#  GuidedFilterCommon.filter computes filtered image for color and gray.
class GuidedFilterCommon:
    def __init__(self, guided_filter):
        self._guided_filter = guided_filter

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        p = np.array(p, dtype=np.float64)
        if _isGray(p):
            return self._filterGray(p)

        cs = p.shape[2]
        q = np.empty_like(p)

        for ci in range(cs):
            q[:, :, ci] = self._filterGray(p[:, :, ci])
        return q

    def _filterGray(self, p):
        ab = self._guided_filter._computeCoefficients(p)
        return self._guided_filter._computeOutput(ab, self._guided_filter._I)


## Guided filter for gray guidance image.
class GuidedFilterGray:
    #  @param I Input gray guidance image.
    #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.4, weights=None):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = np.array(I, dtype=np.float64)
        self.weights = weights
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        I = self._I
        r = self._radius
        if self.weights is None:
            self.weights = 1.0
            self._w_mean = 1.0
        else:
            self._w_mean = boxFilter(self.weights, r)

        w = self.weights

        self._I_mean = boxFilter(I * w, r) / self._w_mean

        self._I_var = (boxFilter(I * I * w, r) - 2 * boxFilter(I * w, r) * self._I_mean  + self._I_mean * self._I_mean * self._w_mean) / self._w_mean + self._epsilon

    def _computeCoefficients(self, p):
        r = self._radius
        I = self._I

        w = self.weights

        p_mean = boxFilter(p * w, r) / self._w_mean

        p_cov = (boxFilter(p * I * w, r) -     boxFilter(p * w, r) * self._I_mean  - boxFilter(I * w, r) * p_mean + p_mean * self._I_mean * self._w_mean) / self._w_mean

        a = p_cov / self._I_var
        b = p_mean - a * self._I_mean
        a_mean = boxFilter(a, r)
        b_mean = boxFilter(b, r)
        return a_mean, b_mean

    def _computeOutput(self, ab, I):
        a_mean, b_mean = ab
        return a_mean * I + b_mean


## Guided filter for color guidance image.
class GuidedFilterColor:
    #  @param I Input color guidance image.
   #  @param radius Radius of Guided Filter.
    #  @param epsilon Regularization term of Guided Filter.
    def __init__(self, I, radius=5, epsilon=0.2, weights = None):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = np.array(I, dtype=np.float64)
        self.weights = weights
        self._initFilter()
        self._filter_common = GuidedFilterCommon(self)

    ## Apply filter for the input image.
    #  @param p Input image for the filtering.
    def filter(self, p):
        return self._filter_common.filter(p)

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        if self.weights is None:
            self.weights = 1.0
            self._w_mean = 1.0
        else:
            self._w_mean = boxFilter(self.weights, r)

        w = self.weights
        print("init_color")
        self._Ir_mean = boxFilter(Ir * w, r) / self._w_mean
        self._Ig_mean = boxFilter(Ig * w, r) / self._w_mean
        self._Ib_mean = boxFilter(Ib * w, r) / self._w_mean

        Irr_var = (boxFilter(Ir * Ir * w, r) - 2 * boxFilter(Ir * w, r) * self._Ir_mean  + self._Ir_mean * self._Ir_mean * self._w_mean) / self._w_mean + eps
        Irg_var = (boxFilter(Ir * Ig * w, r) -     boxFilter(Ir * w, r) * self._Ig_mean  - boxFilter(Ig * w, r) * self._Ir_mean + self._Ir_mean * self._Ig_mean * self._w_mean) / self._w_mean
        Irb_var = (boxFilter(Ir * Ib * w, r) -     boxFilter(Ir * w, r) * self._Ib_mean  - boxFilter(Ib * w, r) * self._Ir_mean + self._Ir_mean * self._Ib_mean * self._w_mean) / self._w_mean
        Igg_var = (boxFilter(Ig * Ig * w, r) - 2 * boxFilter(Ig * w, r) * self._Ig_mean  + self._Ig_mean * self._Ig_mean * self._w_mean) / self._w_mean + eps
        Igb_var = (boxFilter(Ig * Ib * w, r) -     boxFilter(Ig * w, r) * self._Ib_mean  - boxFilter(Ib * w, r) * self._Ig_mean + self._Ig_mean * self._Ib_mean * self._w_mean) / self._w_mean
        Ibb_var = (boxFilter(Ib * Ib * w, r) - 2 * boxFilter(Ib * w, r) * self._Ib_mean  + self._Ib_mean * self._Ib_mean * self._w_mean) / self._w_mean + eps


        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        Irr_inv /= I_cov
        Irg_inv /= I_cov
        Irb_inv /= I_cov
        Igg_inv /= I_cov
        Igb_inv /= I_cov
        Ibb_inv /= I_cov

        self._Irr_inv = Irr_inv
        self._Irg_inv = Irg_inv
        self._Irb_inv = Irb_inv
        self._Igg_inv = Igg_inv
        self._Igb_inv = Igb_inv
        self._Ibb_inv = Ibb_inv

    def _computeCoefficients(self, p):
        r = self._radius
        I = self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        w = self.weights

        p_mean = boxFilter(p * w, r) / self._w_mean

        Ipr_cov = (boxFilter(p * Ir * w, r) -     boxFilter(p * w, r) * self._Ir_mean  - boxFilter(Ir * w, r) * p_mean + p_mean * self._Ir_mean * self._w_mean) / self._w_mean
        Ipg_cov = (boxFilter(p * Ig * w, r) -     boxFilter(p * w, r) * self._Ig_mean  - boxFilter(Ig * w, r) * p_mean + p_mean * self._Ig_mean * self._w_mean) / self._w_mean
        Ipb_cov = (boxFilter(p * Ib * w, r) -     boxFilter(p * w, r) * self._Ib_mean  - boxFilter(Ib * w, r) * p_mean + p_mean * self._Ib_mean * self._w_mean) / self._w_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov
        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar_mean = boxFilter(ar, r)
        ag_mean = boxFilter(ag, r)
        ab_mean = boxFilter(ab, r)
        b_mean = boxFilter(b, r)

        return ar_mean, ag_mean, ab_mean, b_mean

    def _computeOutput(self, ab, I):
        ar_mean, ag_mean, ab_mean, b_mean = ab

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        q = (ar_mean * Ir +
             ag_mean * Ig +
             ab_mean * Ib +
             b_mean)

        return q
