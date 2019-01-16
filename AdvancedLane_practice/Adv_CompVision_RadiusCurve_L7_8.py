import numpy as np

def generate_data(ym_per_pix,xm_per_pix):
    np.random.seed(0)
    ploty = np.linspace(0,719,num=720)
    quadratic_coeff = 3e-4
    leftx = np.array([200+(y**2)*quadratic_coeff+np.random.randint(-50,high=51) for y in ploty])
    rightx = np.array([900+(y**2)*quadratic_coeff+np.random.randint(-50,high=51) for y in ploty])

    leftx = leftx[::1]
    rightx = rightx[::-1]

    left_fit_cr = np.polyfit(ploty*ym_per_pix,leftx*xm_per_pix,2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix,rightx*xm_per_pix,2)

    return ploty,left_fit_cr,right_fit_cr


def measure_curvature_real():

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    ploty,left_fit_cr,right_fit_cr = generate_data(ym_per_pix,xm_per_pix)
    y_eval = np.max(ploty)

    left_curverad = ((1+(2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])

    return left_curverad,right_curverad

left_curverad,right_curverad = measure_curvature_real()

print(left_curverad,'m',right_curverad,'m')
