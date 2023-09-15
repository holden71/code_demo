### Modules
import numpy as np
from scipy.linalg import det, lu
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, eigsh

np.set_printoptions(suppress = True, linewidth=300)

# Graphs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
###


### Main parameters
length_x = 1
length_y = 1000
amount = 101

## TEMP TEMP
fix_index_x = int(amount / 2) - int(amount / 3)
fix_index_y =  int(amount / 2) - int(amount / 5)
## TEMP TEMP

h = 0
frequency = 0

################# МОМЕНТЫ И ТЕМПЕРАТУРЫ ПЕРЕПРОВЕРИТЬ ВО ВСЕХ ФОРМУЛАХ!!!
alpha_x = 0
alpha_y = 0

moment_extra_x = 0
moment_extra_y = 0
temperature_epsilon = 0
###


### Boundary conditions (supported, clamped, free)
boundary_left = 'clamped'
boundary_bottom = 'clamped'
boundary_right = 'clamped'
boundary_top = 'clamped'
###


### Physical parameters
E = 1
v = 0.3
G = E / (2 * (1 + v))
###


### Auxiliary variables
amount_x = amount
amount_y = amount
small_side_x = length_x / amount_x
small_side_y = length_y / amount_y
###

### Anisotropic parameters
soap_elements = []
non_soap_coords = []
# for i in range(int(amount/2)):
#     for j in range(int(amount/2)):
#         soap_elements.append(int(amount**2 / 4 + amount / 4 + j + amount * i))


for index_y in range(amount_y):
    for index_x in range(amount_x):
        index_combined = index_y * amount_x + index_x

        if index_combined not in soap_elements:
            non_soap_coords.append([index_x, index_y])

def D(index_x, index_y):
    index_combined = index_y * amount_x + index_x

    if index_combined in soap_elements:
        return 10**(-10)
    else:
        return 1

def P(index_x, index_y, add_index):
    # index_combined = index_y * amount_x + index_x
    #
    # if index_combined in soap_elements:
    #     return 0
    # else:
    #     if add_index is not None:
    #         if index_combined != add_index:
    #             return 0
    #     return 1

    return 1

### Linear system build
def system_solver(add_index=None, frequency=frequency):
    rows, columns, data = [], [], []
    coefficients_b = np.zeros((amount_x * amount_y * 28), dtype=np.float32)

    for index_y in range(amount_y):
        for index_x in range(amount_x):
            index_combined = index_y * amount_x + index_x

            ## Connectivity equations
            # W_x(a)
            rows.append(0 + index_combined * 28)
            columns.append(12 + index_combined * 28)
            data.append(-1)

            rows.append(0 + index_combined * 28)
            columns.append(0 + index_combined * 28)
            data.append(1)

            rows.append(0 + index_combined * 28)
            columns.append(1 + index_combined * 28)
            data.append(small_side_x)

            rows.append(0 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append((small_side_x**2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(0 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append(small_side_x**3 / (6 * D(index_x, index_y) * (1 - v**2)))

            rows.append(0 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append(-(v * small_side_x**2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(0 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append(-(v * small_side_x**2 * small_side_y) / (4 * D(index_x, index_y) * (1 - v**2)))

            rows.append(0 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append((2 * small_side_x**4 + 3 * small_side_y**2 * v * small_side_x**2) / (48 * D(index_x, index_y) * (1 - v**2)))

            rows.append(0 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append(-(small_side_x**3) / (6 * D(index_x, index_y) * (1 - v**2)))

            rows.append(0 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append((v * small_side_x**2 * small_side_y) / (4 * D(index_x, index_y) * (1 - v**2)))

            rows.append(0 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append(-(frequency**2 * small_side_x**2 * v * small_side_y**2) / (16 * D(index_x, index_y) * (1 - v**2)))

            coefficients_b[0 + index_combined * 28] = (P(index_x, index_y, add_index) * small_side_x**2 * v * small_side_y**2) / (16 * D(index_x, index_y) * (1 - v**2))


            # Theta_xx(a)
            rows.append(1 + index_combined * 28)
            columns.append(13 + index_combined * 28)
            data.append(-1)

            rows.append(1 + index_combined * 28)
            columns.append(1 + index_combined * 28)
            data.append(1)

            rows.append(1 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append(small_side_x / (D(index_x, index_y) * (1 - v**2)))

            rows.append(1 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append((small_side_x ** 2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(1 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append(-(v * small_side_x) / (D(index_x, index_y) * (1 - v**2)))

            rows.append(1 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append(-(v * small_side_x * small_side_y) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(1 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append((4 * small_side_x**3 + 3 * small_side_y**2 * v * small_side_x) / (24 * D(index_x, index_y) * (1 - v**2)))

            rows.append(1 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append(-(small_side_x ** 2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(1 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append((v * small_side_x * small_side_y) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(1 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append(-(frequency**2 * v * small_side_x * small_side_y**2) / (8 * D(index_x, index_y) * (1 - v**2)))

            coefficients_b[1 + index_combined * 28] = (P(index_x, index_y, add_index) * v * small_side_x * small_side_y**2) / (8 * D(index_x, index_y) * (1 - v**2))


            # Theta_xy(a)
            rows.append(2 + index_combined * 28)
            columns.append(14 + index_combined * 28)
            data.append(-1)

            rows.append(2 + index_combined * 28)
            columns.append(2 + index_combined * 28)
            data.append(1)

            rows.append(2 + index_combined * 28)
            columns.append(4 + index_combined * 28)
            data.append(small_side_x / (D(index_x, index_y) * (1 - v)))

            rows.append(2 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append((small_side_x ** 2) / (2 * D(index_x, index_y) * (1 - v)))


            # M_x(a)
            rows.append(3 + index_combined * 28)
            columns.append(15 + index_combined * 28)
            data.append(-1)

            rows.append(3 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append(1)

            rows.append(3 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append(small_side_x)

            rows.append(3 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append((small_side_x ** 2) / 2)

            rows.append(3 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append(-small_side_x)


            # M_xt(a)
            rows.append(4 + index_combined * 28)
            columns.append(16 + index_combined * 28)
            data.append(-1)

            rows.append(4 + index_combined * 28)
            columns.append(4 + index_combined * 28)
            data.append(1)

            rows.append(4 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append(small_side_x)


            # Q_x(a)
            rows.append(5 + index_combined * 28)
            columns.append(17 + index_combined * 28)
            data.append(-1)

            rows.append(5 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append(1)

            rows.append(5 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append(small_side_x)


            # W_y(b)
            rows.append(6 + index_combined * 28)
            columns.append(18 + index_combined * 28)
            data.append(-1)

            rows.append(6 + index_combined * 28)
            columns.append(6 + index_combined * 28)
            data.append(1)

            rows.append(6 + index_combined * 28)
            columns.append(7 + index_combined * 28)
            data.append(small_side_y)

            rows.append(6 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append((small_side_y ** 2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(6 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append((small_side_y**3) / (6 * D(index_x, index_y) * (1 - v**2)))

            rows.append(6 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append(-(v * small_side_y ** 2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(6 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append(-(v * small_side_y ** 2 * small_side_x) / (4 * D(index_x, index_y) * (1 - v**2)))

            rows.append(6 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append(-(2 * small_side_y**4 + 3 * small_side_x**2 * v * small_side_y**2) / (48 * D(index_x, index_y) * (1 - v**2)))

            rows.append(6 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append((v * small_side_x * small_side_y**2) / (4 * D(index_x, index_y) * (1 - v**2)))

            rows.append(6 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append(-(small_side_y**3) / (6 * D(index_x, index_y) * (1 - v**2)))

            rows.append(6 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append((frequency**2 * small_side_y**4) / (24 * D(index_x, index_y) * (1 - v**2)))

            coefficients_b[6 + index_combined * 28] = -(P(index_x, index_y, add_index) * small_side_y**4) / (24 * D(index_x, index_y) * (1 - v**2))


            # Theta_yy(b)
            rows.append(7 + index_combined * 28)
            columns.append(19 + index_combined * 28)
            data.append(-1)

            rows.append(7 + index_combined * 28)
            columns.append(7 + index_combined * 28)
            data.append(1)

            rows.append(7 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append(small_side_y / (D(index_x, index_y) * (1 - v**2)))

            rows.append(7 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append((small_side_y ** 2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(7 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append(-(v * small_side_y) / (D(index_x, index_y) * (1 - v**2)))

            rows.append(7 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append(-(v * small_side_y * small_side_x) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(7 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append(-(4 * small_side_y**3 + 3 * small_side_x**2 * v * small_side_y) / (24 * D(index_x, index_y) * (1 - v**2)))

            rows.append(7 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append((v * small_side_y * small_side_x) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(7 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append(-(small_side_y ** 2) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(7 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append((frequency**2 * small_side_y**3) / (6 * D(index_x, index_y) * (1 - v**2)))

            coefficients_b[7 + index_combined * 28] = -(P(index_x, index_y, add_index) * small_side_y**3) / (6 * D(index_x, index_y) * (1 - v**2))


            # Theta_yx(b)
            rows.append(8 + index_combined * 28)
            columns.append(20 + index_combined * 28)
            data.append(-1)

            rows.append(8 + index_combined * 28)
            columns.append(8 + index_combined * 28)
            data.append(1)

            rows.append(8 + index_combined * 28)
            columns.append(10 + index_combined * 28)
            data.append(small_side_y / (D(index_x, index_y) * (1 - v)))

            rows.append(8 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append((small_side_y ** 2) / (2 * D(index_x, index_y) * (1 - v)))


            # M_y(b)
            rows.append(9 + index_combined * 28)
            columns.append(21 + index_combined * 28)
            data.append(-1)

            rows.append(9 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append(1)

            rows.append(9 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append(small_side_y)

            rows.append(9 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append(-(small_side_y ** 2) / 2)

            rows.append(9 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append(-small_side_y)

            rows.append(9 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append((frequency**2 * small_side_y**2) / 2)

            coefficients_b[9 + index_combined * 28] = -(P(index_x, index_y, add_index) * small_side_y**2 / 2 )


            # M_yt(b)
            rows.append(10 + index_combined * 28)
            columns.append(22 + index_combined * 28)
            data.append(-1)

            rows.append(10 + index_combined * 28)
            columns.append(10 + index_combined * 28)
            data.append(1)

            rows.append(10 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append(small_side_y)


            # Q_y(b)
            rows.append(11 + index_combined * 28)
            columns.append(23 + index_combined * 28)
            data.append(-1)

            rows.append(11 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append(1)

            rows.append(11 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append(-small_side_y)

            rows.append(11 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append(small_side_y * frequency**2)

            coefficients_b[11 + index_combined * 28] = -P(index_x, index_y, add_index) * small_side_y  # Dependent variable


            ### Conjugation & boundary equations
            # LEFT SIDE
            if index_x == 0:
                if boundary_left == 'clamped':
                    columns.append(0 + index_combined * 28)
                    columns.append(1 + index_combined * 28)
                    columns.append(2 + index_combined * 28)
                elif boundary_left == 'supported':
                    columns.append(0 + index_combined * 28)
                    columns.append(2 + index_combined * 28)
                    columns.append(3 + index_combined * 28)
                else:
                    columns.append(3 + index_combined * 28)
                    columns.append(4 + index_combined * 28)
                    columns.append(5 + index_combined * 28)

                rows.append(12 + index_combined * 28)
                data.append(1)

                rows.append(13 + index_combined * 28)
                data.append(1)

                rows.append(14 + index_combined * 28)
                data.append(1)

            else:
                # Equality of M_x
                rows.append(12 + index_combined * 28)
                columns.append(15 + (index_combined - 1) * 28)
                data.append(1)

                rows.append(12 + index_combined * 28)
                columns.append(3 + index_combined * 28)
                data.append(-1)

                # Equality of M_xt
                rows.append(13 + index_combined * 28)
                columns.append(16 + (index_combined - 1) * 28)
                data.append(1)

                rows.append(13 + index_combined * 28)
                columns.append(4 + index_combined * 28)
                data.append(-1)

                # Equality of Q_x
                rows.append(14 + index_combined * 28)
                columns.append(17 + (index_combined - 1) * 28)
                data.append(1)

                rows.append(14 + index_combined * 28)
                columns.append(5 + index_combined * 28)
                data.append(-1)

            # BOTTOM SIDE
            if index_y == 0:
                if boundary_bottom == 'clamped':
                    columns.append(6 + index_combined * 28)
                    columns.append(7 + index_combined * 28)
                    columns.append(8 + index_combined * 28)
                elif boundary_bottom == 'supported':
                    columns.append(6 + index_combined * 28)
                    columns.append(8 + index_combined * 28)
                    columns.append(9 + index_combined * 28)
                else:
                    columns.append(9 + index_combined * 28)
                    columns.append(10 + index_combined * 28)
                    columns.append(11 + index_combined * 28)

                rows.append(15 + index_combined * 28)
                data.append(1)

                rows.append(16 + index_combined * 28)
                data.append(1)

                rows.append(17 + index_combined * 28)
                data.append(1)

            else:
                # Equality of M_y
                rows.append(15 + index_combined * 28)
                columns.append(21 + ((index_y - 1) * amount_x + index_x) * 28)
                data.append(1)

                rows.append(15 + index_combined * 28)
                columns.append(9 + index_combined * 28)
                data.append(-1)

                # Equality of M_yt
                rows.append(16 + index_combined * 28)
                columns.append(22 + ((index_y - 1) * amount_x + index_x) * 28)
                data.append(1)

                rows.append(16 + index_combined * 28)
                columns.append(10 + index_combined * 28)
                data.append(-1)

                # Equality of Q_y
                rows.append(17 + index_combined * 28)
                columns.append(23 + ((index_y - 1) * amount_x + index_x) * 28)
                data.append(1)

                rows.append(17 + index_combined * 28)
                columns.append(11 + index_combined * 28)
                data.append(-1)

            # RIGHT SIDE
            if index_x == (amount_x - 1):
                if boundary_right == 'clamped':
                    columns.append(12 + index_combined * 28)
                    columns.append(13 + index_combined * 28)
                    columns.append(14 + index_combined * 28)
                elif boundary_right == 'supported':
                    columns.append(12 + index_combined * 28)
                    columns.append(14 + index_combined * 28)
                    columns.append(15 + index_combined * 28)
                else:
                    columns.append(15 + index_combined * 28)
                    columns.append(16 + index_combined * 28)
                    columns.append(17 + index_combined * 28)

                rows.append(18 + index_combined * 28)
                data.append(1)

                rows.append(19 + index_combined * 28)
                data.append(1)

                rows.append(20 + index_combined * 28)
                data.append(1)

            else:
                # Equality of W_x
                rows.append(18 + index_combined * 28)
                columns.append(12 + index_combined * 28)
                data.append(1)

                rows.append(18 + index_combined * 28)
                columns.append(0 + (index_combined + 1) * 28)
                data.append(-1)

                # Equality of Theta_xx
                rows.append(19 + index_combined * 28)
                columns.append(13 + index_combined * 28)
                data.append(1)

                rows.append(19 + index_combined * 28)
                columns.append(1 + (index_combined + 1) * 28)
                data.append(-1)

                # Equality of Theta_xy
                rows.append(20 + index_combined * 28)
                columns.append(14 + index_combined * 28)
                data.append(1)

                rows.append(20 + index_combined * 28)
                columns.append(2 + (index_combined + 1) * 28)
                data.append(-1)

            # TOP SIDE
            if index_y == (amount_y - 1):
                if boundary_top == 'clamped':
                    columns.append(18 + index_combined * 28)
                    columns.append(19 + index_combined * 28)
                    columns.append(20 + index_combined * 28)
                elif boundary_top == 'supported':
                    columns.append(18 + index_combined * 28)
                    columns.append(20 + index_combined * 28)
                    columns.append(21 + index_combined * 28)
                else:
                    columns.append(21 + index_combined * 28)
                    columns.append(22 + index_combined * 28)
                    columns.append(23 + index_combined * 28)

                rows.append(21 + index_combined * 28)
                data.append(1)

                rows.append(22 + index_combined * 28)
                data.append(1)

                rows.append(23 + index_combined * 28)
                data.append(1)

            else:
                # Equality of W_y
                rows.append(21 + index_combined * 28)
                columns.append(18 + index_combined * 28)
                data.append(1)

                rows.append(21 + index_combined * 28)
                columns.append(6 + ((index_y + 1) * amount_x + index_x) * 28)
                data.append(-1)

                # Equality of Theta_yy
                rows.append(22 + index_combined * 28)
                columns.append(19 + index_combined * 28)
                data.append(1)

                rows.append(22 + index_combined * 28)
                columns.append(7 + ((index_y + 1) * amount_x + index_x) * 28)
                data.append(-1)

                # Equality of Theta_yx
                rows.append(23 + index_combined * 28)
                columns.append(20 + index_combined * 28)
                data.append(1)

                rows.append(23 + index_combined * 28)
                columns.append(8 + ((index_y + 1) * amount_x + index_x) * 28)
                data.append(-1)


            ## Auxiliary constants
            # Equality: W_x(a/2) = W_c
            rows.append(24 + index_combined * 28)
            columns.append(0 + index_combined * 28)
            data.append(1)

            rows.append(24 + index_combined * 28)
            columns.append(1 + index_combined * 28)
            data.append(small_side_x / 2)

            rows.append(24 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append((small_side_x**2) / (8 * D(index_x, index_y) * (1 - v**2)))

            rows.append(24 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append(small_side_x**3 / (48 * D(index_x, index_y) * (1 - v**2)))

            rows.append(24 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append(-(v * small_side_x**2) / (8 * D(index_x, index_y) * (1 - v**2)))

            rows.append(24 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append(-(v * small_side_x**2 * small_side_y) / (16 * D(index_x, index_y) * (1 - v**2)))

            rows.append(24 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append((small_side_x**4 / 8 + 3 * small_side_y**2 * v * small_side_x**2 / 4) / (48 * D(index_x, index_y) * (1 - v**2)))

            rows.append(24 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append(-(small_side_x**3) / (48 * D(index_x, index_y) * (1 - v**2)))

            rows.append(24 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append((v * small_side_x**2 * small_side_y) / (16 * D(index_x, index_y) * (1 - v**2)))

            rows.append(24 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append(-1 -(frequency**2 * small_side_x**2 * v * small_side_y**2) / (64 * D(index_x, index_y) * (1 - v**2)))

            coefficients_b[24 + index_combined * 28] = (P(index_x, index_y, add_index) * small_side_x**2 * v * small_side_y**2) / (64 * D(index_x, index_y) * (1 - v**2))


            # Equality: Theta_xx(a/2) = Theta_yx(b/2)
            rows.append(25 + index_combined * 28)
            columns.append(1 + index_combined * 28)
            data.append(1)

            rows.append(25 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append(small_side_x / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(25 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append((small_side_x ** 2) / (4 * 2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(25 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append(-(v * small_side_x) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(25 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append(-(v * small_side_x * small_side_y) / (2 * 2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(25 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append((4 * small_side_x**3 / 8 + 3 * small_side_y**2 * v * small_side_x / 2) / (24 * D(index_x, index_y) * (1 - v**2)))

            rows.append(25 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append((-(small_side_x ** 2) / (4 * 2 * D(index_x, index_y) * (1 - v**2))) - ((small_side_y ** 2) / (4 * 2 * D(index_x, index_y) * (1 - v))))

            rows.append(25 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append((v * small_side_x * small_side_y) / (2 * 2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(25 + index_combined * 28)
            columns.append(8 + index_combined * 28)
            data.append(-1)

            rows.append(25 + index_combined * 28)
            columns.append(10 + index_combined * 28)
            data.append(-small_side_y / (2 * D(index_x, index_y) * (1 - v)))

            rows.append(25 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append(-(frequency**2 * v * small_side_x * small_side_y**2) / (2 * 8 * D(index_x, index_y) * (1 - v**2)))

            coefficients_b[25 + index_combined * 28] = (P(index_x, index_y, add_index) * v * small_side_x * small_side_y**2) / (2 * 8 * D(index_x, index_y) * (1 - v**2))


            # Equality: Theta_yy(b/2) = Theta_xy(a/2)
            rows.append(26 + index_combined * 28)
            columns.append(7 + index_combined * 28)
            data.append(1)

            rows.append(26 + index_combined * 28)
            columns.append(9 + index_combined * 28)
            data.append(small_side_y / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(26 + index_combined * 28)
            columns.append(11 + index_combined * 28)
            data.append((small_side_y ** 2) / (4 * 2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(26 + index_combined * 28)
            columns.append(3 + index_combined * 28)
            data.append(-(v * small_side_y) / (2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(26 + index_combined * 28)
            columns.append(5 + index_combined * 28)
            data.append(-(v * small_side_y * small_side_x) / (2 * 2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(26 + index_combined * 28)
            columns.append(24 + index_combined * 28)
            data.append(-(4 * small_side_y**3 + 3 * small_side_x**2 * v * small_side_y) / (8 * 24 * D(index_x, index_y) * (1 - v**2)))

            rows.append(26 + index_combined * 28)
            columns.append(25 + index_combined * 28)
            data.append((v * small_side_y * small_side_x) / (2 * 2 * D(index_x, index_y) * (1 - v**2)))

            rows.append(26 + index_combined * 28)
            columns.append(26 + index_combined * 28)
            data.append((-(small_side_y ** 2) / (4 * 2 * D(index_x, index_y) * (1 - v**2))) - ((small_side_x ** 2) / (4 * 2 * D(index_x, index_y) * (1 - v))))

            rows.append(26 + index_combined * 28)
            columns.append(2 + index_combined * 28)
            data.append(-1)

            rows.append(26 + index_combined * 28)
            columns.append(4 + index_combined * 28)
            data.append(-small_side_x / (2 * D(index_x, index_y) * (1 - v)))

            rows.append(26 + index_combined * 28)
            columns.append(27 + index_combined * 28)
            data.append((frequency**2 * small_side_y**3) / (8 * 6 * D(index_x, index_y) * (1 - v**2)))

            coefficients_b[26 + index_combined * 28] = -(P(index_x, index_y, add_index) * small_side_y**3) / (8 * 6 * D(index_x, index_y) * (1 - v**2))


            # Equality: W_y(b/2) = W_c

            if False:#index_x == fix_index_x and index_y == fix_index_y:
                rows.append(27 + index_combined * 28)
                columns.append(27 + index_combined * 28)
                data.append(1)
                coefficients_b[27 + index_combined * 28] = 1

            else:
                rows.append(27 + index_combined * 28)
                columns.append(6 + index_combined * 28)
                data.append(1)

                rows.append(27 + index_combined * 28)
                columns.append(7 + index_combined * 28)
                data.append(small_side_y / 2)

                rows.append(27 + index_combined * 28)
                columns.append(9 + index_combined * 28)
                data.append((small_side_y ** 2) / (8 * D(index_x, index_y) * (1 - v**2)))

                rows.append(27 + index_combined * 28)
                columns.append(11 + index_combined * 28)
                data.append((small_side_y**3) / (48 * D(index_x, index_y) * (1 - v**2)))

                rows.append(27 + index_combined * 28)
                columns.append(3 + index_combined * 28)
                data.append(-(v * small_side_y ** 2) / (8 * D(index_x, index_y) * (1 - v**2)))

                rows.append(27 + index_combined * 28)
                columns.append(5 + index_combined * 28)
                data.append(-(v * small_side_y ** 2 * small_side_x) / (16 * D(index_x, index_y) * (1 - v**2)))

                rows.append(27 + index_combined * 28)
                columns.append(24 + index_combined * 28)
                data.append(-(small_side_y**4 / 8 + 3 * small_side_x**2 * v * small_side_y**2 / 4) / (48 * D(index_x, index_y) * (1 - v**2)))

                rows.append(27 + index_combined * 28)
                columns.append(25 + index_combined * 28)
                data.append((v * small_side_x * small_side_y**2) / (16 * D(index_x, index_y) * (1 - v**2)))

                rows.append(27 + index_combined * 28)
                columns.append(26 + index_combined * 28)
                data.append(-(small_side_y**3) / (48 * D(index_x, index_y) * (1 - v**2)))

                rows.append(27 + index_combined * 28)
                columns.append(27 + index_combined * 28)
                data.append(-1 + (frequency**2 * small_side_y**4) / (384 * D(index_x, index_y) * (1 - v**2)))

                coefficients_b[27 + index_combined * 28] = -(P(index_x, index_y, add_index) * small_side_y**4) / (384 * D(index_x, index_y) * (1 - v**2))


    sparse_matrix = csc_matrix((data, (rows, columns)), shape=(28 * amount_x * amount_y, 28 * amount_x * amount_y))


    return spsolve(sparse_matrix, coefficients_b), sparse_matrix
###


### Results processing
result, _ = system_solver()
###


### Param functions
def w_x(index_x, index_y, x, result=None, frequency=None, add_index=None):
    temp_array = np.zeros(shape=(28,))
    index_combined = index_y * amount_x + index_x

    temp_array[0] = 1
    temp_array[1] = x
    temp_array[3] = (x**2) / (2 * D(index_x, index_y) * (1 - v**2))
    temp_array[5] = x**3 / (6 * D(index_x, index_y) * (1 - v**2))
    temp_array[9] = -(v * x**2) / (2 * D(index_x, index_y) * (1 - v**2))
    temp_array[11] = -(v * x**2 * small_side_y) / (4 * D(index_x, index_y) * (1 - v**2))
    temp_array[24] = (2 * x**4 + 3 * small_side_y**2 * v * x**2) / (48 * D(index_x, index_y) * (1 - v**2))
    temp_array[25] = -(x**3) / (6 * D(index_x, index_y) * (1 - v**2))
    temp_array[26] = (v * x**2 * small_side_y) / (4 * D(index_x, index_y) * (1 - v**2))
    temp_array[27] = -(frequency**2 * x**2 * v * small_side_y**2) / (16 * D(index_x, index_y) * (1 - v**2))
    dependent_variable = -(P(index_x, index_y, add_index) * x**2 * v * small_side_y**2) / (16 * D(index_x, index_y) * (1 - v**2))

    return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array) + dependent_variable


# def theta_xx(index_x, index_y, x):
#     temp_array = np.zeros(shape=(28,))
#     index_combined = index_y * amount_x + index_x
#
#     temp_array[1] = 1  # Theta_xx_0
#     temp_array[3] = x / (D(index_x, index_y) * (1 - v**2))  # M_x_0
#     temp_array[5] = (x ** 2) / (2 * D(index_x, index_y) * (1 - v**2))  # Q_x_0
#     temp_array[9] = -(v * x) / (D(index_x, index_y) * (1 - v**2))  # M_y_0
#     temp_array[11] = -(v * x * small_side_y) / (2 * D(index_x, index_y) * (1 - v**2))  # Q_y_0
#     temp_array[24] = (x ** 3) / (6 * D(index_x, index_y) * (1 - v**2)) + (v * x * small_side_y ** 2) / (6 * D(index_x, index_y) * (1 - v**2))  # A1
#     temp_array[25] = -(x ** 2) / (2 * D(index_x, index_y) * (1 - v**2))  # A2
#     temp_array[26] = (v * x * small_side_y) / (2 * D(index_x, index_y) * (1 - v**2))  # A3
#     dependent_variable = (alpha_x * x**4 / 24 + moment_extra_x * x**2 / 2  -  v * x * (P(index_x, index_y, add_index) * small_side_y**2 / 6 + alpha_y * small_side_y**3 / 48 + moment_extra_y * small_side_y / 2)) / (D(index_x, index_y) * (1 - v**2)) - temperature_epsilon * x  # Dependent variable
#
#     return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array) + dependent_variable


def m_x(index_x, index_y, x, result=None):
    temp_array = np.zeros(shape=(28,))
    index_combined = index_y * amount_x + index_x

    temp_array[3] = 1
    temp_array[5] = x
    temp_array[24] = (x ** 2) / 2
    temp_array[25] = -x
    dependent_variable = 0

    return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array) + dependent_variable


# def q_x(index_x, index_y, x):
#     temp_array = np.zeros(shape=(28,))
#     index_combined = index_y * amount_x + index_x
#
#     temp_array[5] = 1
#     temp_array[24] = x
#
#     return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array)


def w_y(index_x, index_y, y, result=None, frequency=None, add_index=None):
    temp_array = np.zeros(shape=(28,))
    index_combined = index_y * amount_x + index_x

    temp_array[6] = 1
    temp_array[7] = y
    temp_array[9] = (y ** 2) / (2 * D(index_x, index_y) * (1 - v**2))
    temp_array[11] = (y**3) / (6 * D(index_x, index_y) * (1 - v**2))
    temp_array[3] = -(v * y ** 2) / (2 * D(index_x, index_y) * (1 - v**2))
    temp_array[5] = -(v * y ** 2 * small_side_x) / (4 * D(index_x, index_y) * (1 - v**2))
    temp_array[24] = -(2 * y**4 + 3 * small_side_x**2 * v * y**2) / (48 * D(index_x, index_y) * (1 - v**2))
    temp_array[25] = (v * small_side_x * y**2) / (4 * D(index_x, index_y) * (1 - v**2))
    temp_array[26] = -(y**3) / (6 * D(index_x, index_y) * (1 - v**2))
    temp_array[27] = (frequency**2 * y**4) / (24 * D(index_x, index_y) * (1 - v**2))

    dependent_variable = (P(index_x, index_y, add_index) * y**4) / (24 * D(index_x, index_y) * (1 - v**2))

    return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array) + dependent_variable


# def theta_xy(index_x, index_y, x):
#     temp_array = np.zeros(shape=(27,))
#     index_combined = index_y * amount_x + index_x
#
#     temp_array[2] = 1  # Theta_xy_0
#     temp_array[4] = x / (D(index_x, index_y) * (1 - v))  # M_xt_0
#     temp_array[26] = (x ** 2) / (2 * D(index_x, index_y) * (1 - v))  # A3
#
#     return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array)


def m_y(index_x, index_y, y, result=None, frequency=None, add_index=None):
    temp_array = np.zeros(shape=(28,))
    index_combined = index_y * amount_x + index_x

    temp_array[9] = 1
    temp_array[11] = y
    temp_array[24] = -(y ** 2) / 2
    temp_array[26] = -y
    temp_array[27] = (frequency**2 * y**2) / 2

    dependent_variable = (P(index_x, index_y, add_index) * y**2 / 2)

    return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array) + dependent_variable


# def q_y(index_x, index_y, y, add_index=None):
#     temp_array = np.zeros(shape=(27,))
#     index_combined = index_y * amount_x + index_x
#
#     temp_array[11] = 1
#     temp_array[24] = -y
#     dependent_variable = (P(index_x, index_y, add_index) * y + (alpha_y * y ** 2) / 2)
#
#     return np.matmul(result[0 + 28 * index_combined: 28 + 28 * index_combined], temp_array) + dependent_variable
###


# ### Graphic functions
# def plot_3d_surface():
#     x_points, y_points, z_points = [], [], []
#
#     for index_y in range(amount_y):
#         for index_x in range(amount_x):
#
#             x_points.append(small_side_x / 2 + index_x * small_side_x)
#             y_points.append(0 + index_y * small_side_y)
#             z_points.append(-w_y(index_x, index_y, 0))
#
#             x_points.append(0 + index_x * small_side_x)
#             y_points.append(small_side_y / 2 + index_y * small_side_y)
#             z_points.append(-w_x(index_x, index_y, 0))
#
#             x_points.append(small_side_x / 2 + index_x * small_side_x)
#             y_points.append(small_side_y / 2 + index_y * small_side_y)
#             z_points.append(-w_x(index_x, index_y, small_side_x / 2))
#
#             if index_x == (amount_x - 1):
#                 x_points.append(length_x)
#                 y_points.append(small_side_y / 2 + index_y * small_side_y)
#                 z_points.append(-w_x(index_x, index_y, small_side_x))
#
#             if index_y == (amount_y - 1):
#                 x_points.append(small_side_x / 2 + index_x * small_side_x)
#                 y_points.append(length_y)
#                 z_points.append(-w_y(index_x, index_y, small_side_y))
#
#
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 10))
#
#     surf = ax.plot_trisurf(np.array(x_points), np.array(y_points), np.array(z_points), cmap='viridis', antialiased=False)
#
#     ax.set_aspect('equal')
#     ax.set_zlim(min(z_points) * 2.5, -min(z_points) * 2.5)
#
#     fig.suptitle(f'Деформація пластини під рівномірним навантаженням\nP={1};  v={v};  h={h};  lx={length_x};  ly={length_y}', fontsize=16)
#     ax.set_title(f'(Максимальне відхилення: {round(min(z_points), 6)})')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     plt.show()
# ###


# from scipy.sparse.linalg import splu
# t1 = np.arange(1, 100, 0.1)
# f1 = []
# for point in t1:
#     result1, data, rows, columns = system_solver(frequency=point)
#     mat = csc_matrix((data, (rows, columns)), shape=(28 * amount_x * amount_y, 28 * amount_x * amount_y))
#
#     #f1.append(w_x(int(amount_x/3), int(amount_y/5), small_side_x/2, result=result1))
#     f1.append(det)
#
# plt.plot(t1, f1)
# plt.axhline(y=0, color='r', linestyle='-')
# plt.show()



def find_freq(a, b, step):
    t1 = np.arange(a, b, step)
    f1 = []

    for point in t1:
        result1, sparse_matrix = system_solver(frequency=point)

        f1.append(1 - w_y(fix_index_x, fix_index_y, small_side_y / 2, result=result1, frequency=point))

    plt.plot(t1, f1)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()


#find_freq(13.430, 13.438, .001)
#find_freq(63.684, 63.684, .003)




### RESULTS
print('Переміщення в центральній точці:', round(w_x(int(amount_x/2), int(amount_y/2), small_side_x/2, result=result, frequency=frequency), 8))
#print('Переміщення в центральній точці:', round(w_y(int(amount_x/2), int(amount_y/2), small_side_y/2, result=result, frequency=frequency), 8))

print('Момент X в центральній точці:', round(m_x(int(amount_x/2), int(amount_y/2), small_side_x/2, result=result), 6))
print('Момент Y в центральній точці:', round(m_y(int(amount_x/2), int(amount_y/2), small_side_y/2, result=result, frequency=frequency), 6))


#print('Момент X справа:', round(m_x(0, int(amount_y/2), 0, result=result), 6))


#print('Переміщення в центральній точці:', round(w_y(int(amount_x/2), int(amount_y/2), small_side_y/2), 6))

#print('Переміщення на границе:', round(w_y(int(amount_x/2), int(amount_y/2), small_side_y), 6))
#print('Переміщення в нижньому центрі:', round(w_x(int(amount_x/2 - 1), int(amount_y/8), small_side_x), 6))
#print('Переміщення X:', round(w_x(81, 40, small_side_x) + small_side_y/2 * theta_xy(81, 40, small_side_x), 6))

