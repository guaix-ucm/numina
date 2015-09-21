#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

"""Tests for wavecalibration routines"""

import numpy as np

from numina.array.wavecal.arccalibration import select_data_for_fit
from numina.array.wavecal.arccalibration import gen_triplets_master
from numina.array.wavecal.arccalibration import arccalibration_direct

import pytest

def test__select_data_for_fit(benchmark):
    result1 = 10
    result2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result3 = [900.87172336, 1029.97190349, 1088.24934745, 2032.73591163, 2319.3003487, 2647.40139802, 2796.56881705,
               3031.96484489, 3467.49240452, 3874.34750735]
    result4 = [3803.108, 3828.371, 3839.724, 4019.131, 4071.996, 4131.762, 4158.584, 4200.653, 4277.558, 4348.119]
    result5 = [2.71028128, 1.53826748, 1.51543978, 1.0607126, 1., 1., 1., 1.09576064, 1.57363445, 2.73684352]

    wv_master = np.array(
        [3719.414, 3803.108, 3828.371, 3839.724, 4019.131, 4071.996, 4131.762, 4158.584, 4200.653, 4277.558, 4348.119])
    xpos_arc = np.array(
        [479.80265378, 900.87172336, 1029.97190349, 1088.24934745, 2032.73591163, 2319.3003487, 2647.40139802,
         2796.56881705, 3031.96484489, 3467.49240452, 3874.34750735])
    solution = [{'lineok': False, 'funcost': 3.8643924558710081, 'type': 'P', 'id': 0},
                {'lineok': True, 'funcost': 2.7102812846180808, 'type': 'D', 'id': 1},
                {'lineok': True, 'funcost': 1.5382674829877947, 'type': 'A', 'id': 2},
                {'lineok': True, 'funcost': 1.5154397771500536, 'type': 'A', 'id': 3},
                {'lineok': True, 'funcost': 1.0607125972899634, 'type': 'A', 'id': 4},
                {'lineok': True, 'funcost': 1.0, 'type': 'A', 'id': 5},
                {'lineok': True, 'funcost': 1.0, 'type': 'A', 'id': 6},
                {'lineok': True, 'funcost': 1.0, 'type': 'A', 'id': 7},
                {'lineok': True, 'funcost': 1.0957606427833742, 'type': 'A', 'id': 8},
                {'lineok': True, 'funcost': 1.5736344502886463, 'type': 'D', 'id': 9},
                {'lineok': True, 'funcost': 2.7368435208772128, 'type': 'E', 'id': 10}]

    out1, out2, out3, out4, out5 = benchmark(select_data_for_fit, wv_master, xpos_arc, solution)

    assert out1 == result1
    assert np.allclose(out2, result2)
    assert np.allclose(out3, result3)
    assert np.allclose(out4, result4)
    assert np.allclose(out5, result5)

#
def test__gen_triplets_master(benchmark):
    lst = np.array([3719.414, 3803.108, 3828.371, 3839.724, 4019.131, 4071.996, 4131.762, 4158.584, 4200.653, 4277.558, 4348.119])
    result1 = 165
    result2 = np.array([0.02184328, 0.02527455, 0.0304957, 0.03438084, 0.03742036, 0.04635319,
                        0.04660031, 0.05324692, 0.05951457, 0.06354752, 0.06718397, 0.07106809,
                        0.07686807, 0.07717568, 0.0921053, 0.09395362, 0.10300555, 0.111412,
                        0.11694588, 0.12397103, 0.13312126, 0.13617566, 0.14995055, 0.16068975,
                        0.16950047, 0.17330385, 0.17391359, 0.18396938, 0.19057313, 0.19136161,
                        0.19521306, 0.20296934, 0.20456454, 0.21555369, 0.21644702, 0.221959,
                        0.22640933, 0.23737457, 0.24809755, 0.25000052, 0.26423555, 0.27394858,
                        0.27924342, 0.2907444, 0.29123192, 0.29176812, 0.30902599, 0.31358489,
                        0.31841355, 0.34122559, 0.34235595, 0.35288899, 0.35359827, 0.36353293,
                        0.36702402, 0.37908829, 0.38933968, 0.39636448, 0.401412, 0.40976032,
                        0.42122571, 0.42388476, 0.42467836, 0.43583294, 0.45531247, 0.4568731,
                        0.46453749, 0.4659409, 0.46873677, 0.46936456, 0.47251639, 0.4767212,
                        0.49336252, 0.49707006, 0.51240726, 0.52151004, 0.53050243, 0.53698866,
                        0.53962241, 0.54236877, 0.54339257, 0.55175873, 0.56080674, 0.56265132,
                        0.56673622, 0.5744313, 0.57768773, 0.58372711, 0.60302269, 0.60770066,
                        0.61432759, 0.62048126, 0.62280281, 0.6258793, 0.62718949, 0.6277152,
                        0.62875959, 0.63170436, 0.63533289, 0.64353931, 0.65223638, 0.65440983,
                        0.65586881, 0.65729612, 0.66700622, 0.67301429, 0.67386773, 0.67542249,
                        0.67637123, 0.68246237, 0.68994429, 0.69023421, 0.69270524, 0.6956529,
                        0.69853111, 0.70241113, 0.70993814, 0.71627404, 0.7268545, 0.72826688,
                        0.72844509, 0.72942564, 0.73265467, 0.73513481, 0.73778137, 0.73878426,
                        0.74445809, 0.74923807, 0.75641675, 0.76544484, 0.76813789, 0.76824297,
                        0.77240046, 0.78300667, 0.78552105, 0.78683996, 0.79534855, 0.80283717,
                        0.80300668, 0.80339398, 0.8076628, 0.80912867, 0.81494942, 0.81814918,
                        0.82435124, 0.82670893, 0.82879068, 0.83790705, 0.85006325, 0.85505932,
                        0.8568466, 0.86120831, 0.86221298, 0.86423998, 0.87053289, 0.88344245,
                        0.88699695, 0.88776771, 0.89417802, 0.90563544, 0.9125819, 0.91588158,
                        0.91877364, 0.92454624, 0.9389257])
    result3 = [(2, 3, 10), (2, 3, 9), (2, 3, 8), (2, 3, 7), (2, 3, 6), (1, 2, 10), (2, 3, 5), (1, 2, 9), (2, 3, 4),
               (1, 2, 8), (1, 3, 10), (1, 2, 7), (1, 2, 6), (1, 3, 9), (1, 3, 8), (1, 2, 5), (1, 3, 7), (1, 3, 6),
               (1, 2, 4), (6, 7, 10), (0, 1, 10), (1, 3, 5), (0, 1, 9), (4, 5, 10), (1, 3, 4), (0, 2, 10), (0, 1, 8),
               (6, 7, 9), (0, 1, 7), (0, 3, 10), (0, 2, 9), (0, 1, 6), (4, 5, 9), (0, 3, 9), (5, 6, 10), (7, 8, 10),
               (0, 2, 8), (0, 1, 5), (0, 2, 7), (0, 3, 8), (0, 2, 6), (0, 3, 7), (0, 1, 4), (5, 6, 9), (4, 5, 8),
               (0, 3, 6), (0, 2, 5), (5, 7, 10), (6, 8, 10), (0, 3, 5), (4, 6, 10), (3, 4, 10), (7, 8, 9), (0, 2, 4),
               (2, 4, 10), (4, 5, 7), (6, 7, 8), (1, 4, 10), (0, 3, 4), (3, 4, 9), (5, 7, 9), (4, 7, 10), (2, 4, 9),
               (4, 6, 9), (1, 4, 9), (3, 5, 10), (5, 6, 8), (5, 8, 10), (2, 5, 10), (4, 5, 6), (6, 8, 9), (0, 4, 10),
               (1, 5, 10), (3, 4, 8), (2, 4, 8), (8, 9, 10), (3, 5, 9), (0, 4, 9), (4, 7, 9), (2, 5, 9), (1, 4, 8),
               (4, 8, 10), (0, 5, 10), (3, 4, 7), (1, 5, 9), (3, 6, 10), (2, 4, 7), (2, 6, 10), (1, 6, 10), (1, 4, 7),
               (3, 4, 6), (4, 6, 8), (0, 4, 8), (5, 8, 9), (3, 7, 10), (7, 9, 10), (2, 4, 6), (0, 5, 9), (2, 7, 10),
               (3, 5, 8), (1, 7, 10), (2, 5, 8), (0, 6, 10), (1, 4, 6), (3, 6, 9), (5, 7, 8), (6, 9, 10), (2, 6, 9),
               (1, 5, 8), (0, 4, 7), (1, 2, 3), (5, 6, 7), (1, 6, 9), (0, 1, 3), (0, 7, 10), (4, 8, 9), (3, 8, 10),
               (2, 8, 10), (0, 4, 6), (3, 7, 9), (3, 5, 7), (1, 8, 10), (0, 5, 8), (2, 7, 9), (2, 5, 7), (0, 6, 9),
               (5, 9, 10), (1, 7, 9), (1, 5, 7), (0, 8, 10), (0, 1, 2), (4, 7, 8), (3, 4, 5), (2, 4, 5), (4, 9, 10),
               (0, 7, 9), (3, 5, 6), (0, 5, 7), (2, 5, 6), (1, 4, 5), (4, 6, 7), (3, 6, 8), (2, 6, 8), (1, 5, 6),
               (3, 8, 9), (1, 6, 8), (2, 8, 9), (1, 8, 9), (0, 4, 5), (0, 5, 6), (0, 6, 8), (3, 9, 10), (0, 8, 9),
               (2, 9, 10), (1, 9, 10), (3, 7, 8), (2, 7, 8), (0, 9, 10), (1, 7, 8), (0, 2, 3), (0, 7, 8), (3, 6, 7),
               (2, 6, 7), (1, 6, 7), (0, 6, 7)]

    out1, out2, out3 = benchmark(gen_triplets_master, lst)

    assert out1 == result1
    assert np.allclose(out2, result2)
    assert out3 == result3


def test__arccalibration_direct(benchmark):
    wv_master = np.array(
        [3719.414, 3803.108, 3828.371, 3839.724, 4019.131, 4071.996, 4131.762, 4158.584, 4200.653, 4277.558, 4348.119])
    ntriplets_master = 165
    ratios_master_sorted = np.array(
        [0.021843277896211384, 0.0252745515787413, 0.030495699496618317, 0.03438083903419936, 0.037420358547221505,
         0.046353192871336435, 0.04660030785017985, 0.053246917483401685, 0.05951457328580457, 0.06354752292193315,
         0.06718396509428254, 0.07106808898491021, 0.07686807402313665, 0.07717567709979976, 0.09210529625576973,
         0.09395361637559105, 0.10300554749125122, 0.11141200167957803, 0.11694588076269637, 0.12397102936350624,
         0.13312125718739318, 0.13617565677903065, 0.14995055039559682, 0.16068975160188292, 0.1695004698573765,
         0.1733038547490476, 0.17391358555728018, 0.18396938187604644, 0.19057312657968448, 0.19136160838549088,
         0.1952130632954935, 0.20296933657978228, 0.20456453853506099, 0.21555369223712875, 0.21644701817668108,
         0.22195900493312817, 0.2264093309145765, 0.2373745681855568, 0.24809754764669711, 0.25000051949239344,
         0.2642355486142773, 0.2739485848304758, 0.27924341962584726, 0.2907443982837278, 0.2912319167924556,
         0.2917681181914308, 0.3090259854445205, 0.3135848878941626, 0.31841354797857496, 0.3412255872392805,
         0.3423559521927849, 0.3528889937941952, 0.3535982651671824, 0.3635329327332117, 0.36702401933244555,
         0.3790882949811065, 0.38933968152588744, 0.39636447704725203, 0.40141199865206206, 0.40976032012132396,
         0.4212257129235938, 0.42388476175422823, 0.42467836335423736, 0.43583294315222404, 0.45531246706712986,
         0.4568731006402505, 0.46453749115865867, 0.4659409031482359, 0.46873677243587314, 0.4693645621542941,
         0.4725163927679799, 0.47672119674569136, 0.4933625192886018, 0.4970700608706966, 0.5124072611622363,
         0.5215100429929613, 0.5305024278607875, 0.5369886624240335, 0.5396224078753378, 0.5423687684639138,
         0.5433925719100974, 0.5517587267620718, 0.5608067376591568, 0.5626513203286705, 0.5667362208873433,
         0.5744312984982146, 0.5776877348862701, 0.583727113909048, 0.6030226912851298, 0.6077006605227917,
         0.6143275875057355, 0.6204812639790197, 0.6228028069213003, 0.6258792967571838, 0.6271894884882817,
         0.6277151977207389, 0.6287595874630427, 0.6317043630317625, 0.6353328920938609, 0.6435393110556367,
         0.6522363768804665, 0.6544098291080415, 0.6558688096961212, 0.6572961229743134, 0.6670062169680739,
         0.6730142938199992, 0.6738677278756883, 0.6754224855127146, 0.676371228414393, 0.6824623722021083,
         0.6899442866506427, 0.6902342125929667, 0.6927052376435866, 0.6956528966835674, 0.6985311075941817,
         0.702411125772463, 0.7099381386520333, 0.7162740404965492, 0.7268545015375363, 0.7282668774010237,
         0.728445085617513, 0.7294256446200176, 0.732654668470344, 0.7351348102238039, 0.7377813714178431,
         0.7387842563926149, 0.7444580857081815, 0.7492380651280426, 0.7564167482474209, 0.7654448429708693,
         0.768137889259066, 0.7682429677945355, 0.7724004615278628, 0.7830066700872232, 0.7855210524396035,
         0.7868399552803574, 0.7953485505310962, 0.8028371701163562, 0.8030066811474312, 0.8033939781619104,
         0.8076627967845789, 0.8091286651945382, 0.814949420063284, 0.8181491781630539, 0.824351238140483,
         0.8267089260335295, 0.8287906818318433, 0.8379070502687327, 0.850063247698407, 0.855059318827787,
         0.8568465980521103, 0.8612083124342298, 0.8622129772961821, 0.8642399778354131, 0.870532888327025,
         0.8834424499001178, 0.8869969539220258, 0.8877677129973521, 0.8941780175829142, 0.9056354417754129,
         0.912581897975849, 0.9158815781220594, 0.9187736400444557, 0.9245462422216967, 0.9389257007536942])
    triplets_master_sorted_list = [(2, 3, 10), (2, 3, 9), (2, 3, 8), (2, 3, 7), (2, 3, 6), (1, 2, 10), (2, 3, 5),
                                   (1, 2, 9), (2, 3, 4), (1, 2, 8), (1, 3, 10), (1, 2, 7), (1, 2, 6), (1, 3, 9),
                                   (1, 3, 8), (1, 2, 5), (1, 3, 7), (1, 3, 6), (1, 2, 4), (6, 7, 10), (0, 1, 10),
                                   (1, 3, 5), (0, 1, 9), (4, 5, 10), (1, 3, 4), (0, 2, 10), (0, 1, 8), (6, 7, 9),
                                   (0, 1, 7), (0, 3, 10), (0, 2, 9), (0, 1, 6), (4, 5, 9), (0, 3, 9), (5, 6, 10),
                                   (7, 8, 10), (0, 2, 8), (0, 1, 5), (0, 2, 7), (0, 3, 8), (0, 2, 6), (0, 3, 7),
                                   (0, 1, 4), (5, 6, 9), (4, 5, 8), (0, 3, 6), (0, 2, 5), (5, 7, 10), (6, 8, 10),
                                   (0, 3, 5), (4, 6, 10), (3, 4, 10), (7, 8, 9), (0, 2, 4), (2, 4, 10), (4, 5, 7),
                                   (6, 7, 8), (1, 4, 10), (0, 3, 4), (3, 4, 9), (5, 7, 9), (4, 7, 10), (2, 4, 9),
                                   (4, 6, 9), (1, 4, 9), (3, 5, 10), (5, 6, 8), (5, 8, 10), (2, 5, 10), (4, 5, 6),
                                   (6, 8, 9), (0, 4, 10), (1, 5, 10), (3, 4, 8), (2, 4, 8), (8, 9, 10), (3, 5, 9),
                                   (0, 4, 9), (4, 7, 9), (2, 5, 9), (1, 4, 8), (4, 8, 10), (0, 5, 10), (3, 4, 7),
                                   (1, 5, 9), (3, 6, 10), (2, 4, 7), (2, 6, 10), (1, 6, 10), (1, 4, 7), (3, 4, 6),
                                   (4, 6, 8), (0, 4, 8), (5, 8, 9), (3, 7, 10), (7, 9, 10), (2, 4, 6), (0, 5, 9),
                                   (2, 7, 10), (3, 5, 8), (1, 7, 10), (2, 5, 8), (0, 6, 10), (1, 4, 6), (3, 6, 9),
                                   (5, 7, 8), (6, 9, 10), (2, 6, 9), (1, 5, 8), (0, 4, 7), (1, 2, 3), (5, 6, 7),
                                   (1, 6, 9), (0, 1, 3), (0, 7, 10), (4, 8, 9), (3, 8, 10), (2, 8, 10), (0, 4, 6),
                                   (3, 7, 9), (3, 5, 7), (1, 8, 10), (0, 5, 8), (2, 7, 9), (2, 5, 7), (0, 6, 9),
                                   (5, 9, 10), (1, 7, 9), (1, 5, 7), (0, 8, 10), (0, 1, 2), (4, 7, 8), (3, 4, 5),
                                   (2, 4, 5), (4, 9, 10), (0, 7, 9), (3, 5, 6), (0, 5, 7), (2, 5, 6), (1, 4, 5),
                                   (4, 6, 7), (3, 6, 8), (2, 6, 8), (1, 5, 6), (3, 8, 9), (1, 6, 8), (2, 8, 9),
                                   (1, 8, 9), (0, 4, 5), (0, 5, 6), (0, 6, 8), (3, 9, 10), (0, 8, 9), (2, 9, 10),
                                   (1, 9, 10), (3, 7, 8), (2, 7, 8), (0, 9, 10), (1, 7, 8), (0, 2, 3), (0, 7, 8),
                                   (3, 6, 7), (2, 6, 7), (1, 6, 7), (0, 6, 7)]
    xpeaks_refined = np.array(
        [479.80265378018044, 900.8717233647345, 1029.9719034878276, 1088.249347453824, 2032.7359116278196,
         2319.3003486972016, 2647.4013980185764, 2796.5688170453263, 3031.964844893816, 3467.492404519348,
         3874.347507353127])
    naxis1 = 4096
    wv_ini_search = 3500
    wv_end_search = 4500
    error_xpos_arc = 2.0
    times_sigma_r = 3.0
    frac_triplets_for_sum = 0.50
    times_sigma_TheilSen = 10.0
    poly_degree = 2
    times_sigma_polfilt = 10.0
    times_sigma_inclusion = 5.0

    result = [{'lineok': False, 'funcost': 3.8643924558710081, 'type': 'P', 'id': 0},
              {'lineok': True, 'funcost': 2.7102812846180808, 'type': 'D', 'id': 1},
              {'lineok': True, 'funcost': 1.5382674829877947, 'type': 'A', 'id': 2},
              {'lineok': True, 'funcost': 1.5154397771500536, 'type': 'A', 'id': 3},
              {'lineok': True, 'funcost': 1.0607125972899634, 'type': 'A', 'id': 4},
              {'lineok': True, 'funcost': 1.0, 'type': 'A', 'id': 5},
              {'lineok': True, 'funcost': 1.0, 'type': 'A', 'id': 6},
              {'lineok': True, 'funcost': 1.0, 'type': 'A', 'id': 7},
              {'lineok': True, 'funcost': 1.0957606427833742, 'type': 'A', 'id': 8},
              {'lineok': True, 'funcost': 1.5736344502886463, 'type': 'D', 'id': 9},
              {'lineok': True, 'funcost': 2.7368435208772128, 'type': 'E', 'id': 10}]

    solution = benchmark(arccalibration_direct,
                         wv_master, ntriplets_master, ratios_master_sorted,
                         triplets_master_sorted_list, xpeaks_refined, naxis1,
                         wv_ini_search, wv_end_search, error_xpos_arc, times_sigma_r,
                         frac_triplets_for_sum, times_sigma_TheilSen, poly_degree,
                         times_sigma_polfilt, times_sigma_inclusion)

    for elem in range(len(solution)):
        for key in solution[elem].keys():
            assert solution[elem][key] == result[elem][key]


# These are the values used for the plots
cdelt1_search = np.array([0.19804269,0.19541287,0.19023419,0.0885073,0.14745287,0.1886778,0.19790001,0.18323951,0.07671992,0.1814241,0.17914079,0.17732869,0.17505821])
cdelt1_search_norm = np.array([ 0.81098483,0.80021572,0.77900901,0.36243739,0.60381949,0.77263561,0.81040052,0.75036578,0.31416808,0.74293167,0.73358154,0.72616098,0.71686336])
cdelt1_layered_list = np.array([np.array([ 0.81098483]), np.array([ 0.80021572]), np.array([ 0.77900901]), np.array([ 0.36243739,  0.60381949,  0.77263561,  0.81040052]), np.array([ 0.75036578]), np.array([ 0.31416808,  0.74293167]), np.array([ 0.73358154]), np.array([ 0.72616098]), np.array([ 0.71686336])])

crval1_search = np.array([3624.89498025,3627.29664333,3632.89199981,3623.28454061,3858.99871411,3635.7875291,3617.05045178,3647.19178965,3625.3392893,3651.64102104,3657.78359128,3663.17597849,3670.7200489 ])
crval1_search_norm = np.array( [ 0.12489498,0.12729664,0.132892, 0.12328454,0.35899871,0.13578753,0.11705045,0.14719179,0.12533929,0.15164102,0.15778359,0.16317598,0.17072005])
crval1_layered_list = np.array([np.array([ 0.12489498]), np.array([ 0.12729664]), np.array([ 0.132892]), np.array([ 0.12328454,  0.35899871,  0.13578753,  0.11705045]), np.array([ 0.14719179]), np.array([ 0.12533929,  0.15164102]), np.array([ 0.15778359]), np.array([ 0.16317598]), np.array([ 0.17072005])])

itriplet_search = np.array( [ 0., 1., 2., 3., 3., 3., 3., 4., 5., 5., 6., 7., 8.])

error_cdelt1_search_norm = np.array([ 0.00416928,0.01207909,0.0021973,0.00083273,0.00138732,0.00177519,0.00186195,0.00345286,0.00186185,0.00440282,0.00539542,0.00306129,0.00240698])
error_crval1_search_norm = np.array([-0.00099815,-0.00306024,-0.0006965,-0.00044948,-0.00074883,-0.00095819,-0.00100503,-0.00198882,-0.00121297,-0.00286837,-0.00370072,-0.00229344,-0.00206741])

clabel_search = np.array(['0,1,2', '1,2,3', '2,3,4', '0,1,2', '4,7,8', '3,4,5', '2,4,5', '4,5,6', '1,2,3', '5,6,7', '6,7,8', '7,8,9', '8,9,10'])

funcost_search = np.array([3.86439246,2.71028128,1.53826748,198.47263166,103.00157997,1.51543978,4.21534537,1.0607126,252.68471095,1.,1.09576064,1.57363445,2.73684352])
funcost_layered_list = np.array([np.array([ 3.86439246]), np.array([ 2.71028128]), np.array([ 1.53826748]), np.array([ 198.47263166,  103.00157997,    1.51543978,    4.21534537]), np.array([ 1.0607126]), np.array([ 252.68471095,    1.        ]), np.array([ 1.09576064]), np.array([ 1.57363445]), np.array([ 2.73684352])])

xmin = -0.05
xmax =  1.05
ymin = -0.05
ymax =  1.05
xp_limits = np.array([0.,1.,0.,0.])
yp_limits = np.array([1.,0.,0.,1.])

@pytest.mark.skipif("True")
def test_gen_triplets_master_plot():
    import matplotlib.pyplot as plt

    ratios_master = np.array([ 0.76813789,0.6956529,0.27924342,0.23737457,0.20296934,0.19057313,0.17391359,0.14995055,0.13312126,0.90563544,0.36353293,0.30902599,0.26423555,0.24809755,0.22640933,0.19521306,0.17330385,0.401412,0.34122559,0.29176812,0.27394858,0.25000052,0.21555369,0.19136161,0.85006325,0.7268545,0.68246237,0.62280281,0.53698866,0.4767212,0.85505932,0.80283717,0.73265467,0.63170436,0.56080674,0.9389257,0.8568466,0.73878426,0.65586881,0.9125819,0.78683996,0.69853111,0.86221298,0.76544484,0.88776771,0.68994429,0.11694588,0.09395362,0.07686807,0.07106809,0.06354752,0.05324692,0.04635319,0.16950047,0.13617566,0.111412,0.10300555,0.0921053,0.07717568,0.06718397,0.80339398,0.65729612,0.60770066,0.54339257,0.45531247,0.39636448,0.81814918,0.75641675,0.67637123,0.56673622,0.49336252,0.92454624,0.82670893,0.69270524,0.60302269,0.89417802,0.74923807,0.65223638,0.83790705,0.72942564,0.87053289,0.05951457,0.04660031,0.03742036,0.03438084,0.0304957,0.02527455,0.02184328,0.78300667,0.62875959,0.57768773,0.51240726,0.42467836,0.36702402,0.80300668,0.73778137,0.65440983,0.54236877,0.46873677,0.91877364,0.81494942,0.67542249,0.58372711,0.88699695,0.73513481,0.63533289,0.82879068,0.71627404,0.86423998,0.77240046,0.61432759,0.56265132,0.49707006,0.40976032,0.35288899,0.79534855,0.72844509,0.64353931,0.53050243,0.4568731,0.91588158,0.80912867,0.66700622,0.5744313,0.88344245,0.72826688,0.62718949,0.82435124,0.70993814,0.86120831,0.46936456,0.37908829,0.29123192,0.20456454,0.16068975,0.8076628,0.62048126,0.43583294,0.34235595,0.76824297,0.53962241,0.42388476,0.70241113,0.55175873,0.78552105,0.69023421,0.46453749,0.2907444,0.21644702,0.67301429,0.42122571,0.31358489,0.6258793,0.4659409,0.74445809,0.38933968,0.18396938,0.12397103,0.47251639,0.31841355,0.67386773,0.35359827,0.221959,0.6277152,0.52151004])
    bins_in = np.linspace(0.0, 1.0, 41)
    hist = [ 1,6,5,4,4,3,4,4,5,3,3,4,3,2,4,3,5,1,7,3,2,5,5,2,5,7,7,8,3,10,5,4,8,3,7,4,5,1,0,0]
    bins_out = np.array([ 0.,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975,1.])

    hist_ori, bins_out_ori = np.histogram(ratios_master, bins=bins_in)

    np.testing.assert_array_almost_equal(bins_out,bins_out_ori)
    np.testing.assert_array_equal(hist_ori,hist_ori)

    width_hist = 0.02
    width_hist_ori = 0.8*(bins_out_ori[1]-bins_out_ori[0])

    np.testing.assert_almost_equal(width_hist, width_hist_ori)

    center = np.array([ 0.0125,0.0375,0.0625,0.0875,0.1125,0.1375,0.1625,0.1875,0.2125,0.2375,0.2625,0.2875,0.3125,0.3375,0.3625,0.3875,0.4125,0.4375,0.4625,0.4875,0.5125,0.5375,0.5625,0.5875,0.6125,0.6375,0.6625,0.6875,0.7125,0.7375,0.7625,0.7875,0.8125,0.8375,0.8625,0.8875,0.9125,0.9375,0.9625,0.9875])
    center_ori = (bins_out_ori[:-1]+bins_out_ori[1:])/2

    np.testing.assert_almost_equal(center, center_ori)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(center, hist, align='center', width=width_hist)
    ax.set_xlabel('distance ratio in each triplet')
    ax.set_ylabel('Number of triplets')
    plt.show()


@pytest.mark.skipif("True")
def arccalibration_direct_plot1():
    # CDELT1 vs CRVAL1 diagram (original coordinates)
    import matplotlib.pyplot as plt

    cdelt1_max = 0.2442002442
    naxis1_arc = 4096
    wv_end_search = 4500
    wv_ini_search = 3500

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('cdelt1 (Angstroms/pixel)')
    ax.set_ylabel('crval1 (Angstroms)')
    ax.scatter(cdelt1_search, crval1_search, s=200, alpha=0.1)
    xmin = 0.0
    xmax = cdelt1_max
    dx = xmax-xmin
    xmin -= dx/20
    xmax += dx/20
    ax.set_xlim([xmin,xmax])
    ymin = wv_ini_search
    ymax = wv_end_search
    dy = ymax-ymin
    ymin -= dy/20
    ymax += dy/20
    ax.set_ylim([ymin,ymax])
    xp_limits = np.array([0., cdelt1_max])
    yp_limits = wv_end_search-float(naxis1_arc-1)*xp_limits
    xp_limits = np.concatenate((xp_limits,[xp_limits[0],xp_limits[0]]))
    yp_limits = np.concatenate((yp_limits,[yp_limits[1],yp_limits[0]]))
    ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
    plt.show()
    print('Number of points in last plot:', len(cdelt1_search))

@pytest.mark.skipif("True")
def arccalibration_direct_plot2():
    # CDELT1 vs CRVAL1 diagram (normalized coordinates)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('normalized cdelt1')
    ax.set_ylabel('normalized crval1')
    ax.scatter(cdelt1_search_norm, crval1_search_norm, s=200, alpha=0.1)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
    plt.show()
    print('Number of points in last plot:', len(cdelt1_search))


@pytest.mark.skipif("True")
def arccalibration_direct_plot3():
    import matplotlib.pyplot as plt
    # CDELT1 vs CRVAL1 diagram (normalized coordinates) with different color for each arc triplet and overplotting
    # the arc number

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('normalized cdelt1')
    ax.set_ylabel('normalized crval1')
    ax.scatter(cdelt1_search_norm, crval1_search_norm, s=200, alpha=0.1, c=itriplet_search)
    for i in range(len(itriplet_search)):
        ax.text(cdelt1_search_norm[i], crval1_search_norm[i], str(int(itriplet_search[i])), fontsize=6)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
    plt.show()


@pytest.mark.skipif("True")
def arccalibration_direct_plot4():
    import matplotlib.pyplot as plt
    # CDELT1 vs CRVAL1 diagram (normalized coordinates)
    # including triplet numbers

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('normalized cdelt1')
    ax.set_ylabel('normalized crval1')
    ax.scatter(cdelt1_search_norm, crval1_search_norm, s=200, alpha=0.1,c=itriplet_search)
    for i in range(len(clabel_search)):
        ax.text(cdelt1_search_norm[i], crval1_search_norm[i],clabel_search[i], fontsize=6)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
    plt.show()


@pytest.mark.skipif("True")
def arccalibration_direct_plot5():
    import matplotlib.pyplot as plt
    # CDELT1 vs CRVAL1 diagram (normalized coordinates) with error bars (note that errors in this plot are highly correlated)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('normalized cdelt1')
    ax.set_ylabel('normalized crval1')
    ax.errorbar(cdelt1_search_norm, crval1_search_norm, xerr=error_cdelt1_search_norm, yerr=error_crval1_search_norm,fmt='none')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
    plt.show()


@pytest.mark.skipif("True")
def arccalibration_direct_plot6():
    import matplotlib.pyplot as plt
    # CDELT1 vs CRVAL1 diagram (normalized coordinates)
    # with symbol size proportional to the inverse of the cost function

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('normalized cdelt1')
    ax.set_ylabel('normalized crval1')
    ax.scatter(cdelt1_search_norm, crval1_search_norm, s=2000/funcost_search, c=itriplet_search, alpha=0.2)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
    plt.show()
    print('Number of points in last plot:', len(cdelt1_search))


@pytest.mark.skipif("True")
def arccalibration_direct_plot7():
    import matplotlib.pyplot as plt
    # CDELT1 vs CRVAL1 diagram (normalized coordinates) with symbol size proportional to the inverse of the cost function
    # and overplotting triplet number

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('normalized cdelt1')
    ax.set_ylabel('normalized crval1')
    ax.scatter(cdelt1_search_norm, crval1_search_norm, s=2000/funcost_search, c=itriplet_search, alpha=0.2)
    for i in range(len(itriplet_search)):
        ax.text(cdelt1_search_norm[i], crval1_search_norm[i], str(int(itriplet_search[i])), fontsize=6)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
    plt.show()
    print('Number of points in last plot:', len(cdelt1_search))


@pytest.mark.skipif("True")
def arccalibration_direct_plot8():
    import matplotlib.pyplot as plt
    # CDELT1 vs CRVAL1 diagram (normalized coordinates)

    ntriplets_arc = 9


    for i in range(ntriplets_arc):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('normalized cdelt1')
        ax.set_ylabel('normalized crval1')
        xdum = cdelt1_layered_list[i]
        ydum = crval1_layered_list[i]
        sdum = 2000/funcost_layered_list[i]
        ax.scatter(xdum, ydum, s=sdum, alpha=0.8)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_title('Arc triplet number '+str(i))
        ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
        plt.show(block=False)
        print('Number of points in last plot:', xdum.size)

if __name__ == '__main__':
    test_gen_triplets_master_plot()
    arccalibration_direct_plot1()
    arccalibration_direct_plot2()
    arccalibration_direct_plot3()
    arccalibration_direct_plot4()
    arccalibration_direct_plot5()
    arccalibration_direct_plot6()
    arccalibration_direct_plot7()
    arccalibration_direct_plot8()