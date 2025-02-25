import scipy.spatial.transform
import numpy as np
from animate_function import QuadPlotter


import csv

# Initialize logging file with headers
# LOG_FILE0 = "custom_figure8_baseline_nowind.csv"
# LOG_FILE1 = "custom_figure8_baseline_10wind.csv"
# LOG_FILE2 = "custom_figure8_baseline_20wind.csv"

# LOG_FILE3 = "custom_spiral_baseline_nowind.csv"
# LOG_FILE4 = "custom_spiral_baseline_10wind.csv"
# LOG_FILE5 = "custom_spiral_baseline_20wind.csv"
# LOG_FILE5 = "custom_spiral_baseline_30wind.csv"

# LOG_FILE6 = "custom_circle_baseline_nowind.csv"
# LOG_FILE6 = "custom_circle_baseline_10wind.csv"
# LOG_FILE6 = "custom_circle_baseline_20wind.csv"
LOG_FILE6 = "custom_circle_baseline_30wind.csv"
# LOG_FILE10 = "custom_random2_baseline_40wind.csv"

# LOG_FILE11= "custom_hover_baseline_nowind.csv"
# LOG_FILE12 ="custom_hover_baseline_10wind.csv"
# LOG_FILE13 = "custom_hover_baseline_20wind.csv"
# LOG_FILE14 = "custom_hover_baseline_30wind.csv"
# LOG_FILE15 = "custom_hover_baseline_40wind.csv"

with open(LOG_FILE6, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["t", "p", "p_d" , "v" , "v_d", "q", "R", "w","T_sp", "q_sp", "hover_throttle","fa", "pwm"])



def quat_mult(q, p):
    # q * p
    # p,q = [w x y z]
    return np.array(
        [
            p[0] * q[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[1] * p[0] + q[0] * p[1] + q[2] * p[3] - q[3] * p[2],
            q[2] * p[0] + q[0] * p[2] + q[3] * p[1] - q[1] * p[3],
            q[3] * p[0] + q[0] * p[3] + q[1] * p[2] - q[2] * p[1],
        ]
    )
    
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_from_vectors(v_from, v_to):
    v_from = normalized(v_from)
    v_to = normalized(v_to)
    v_mid = normalized(v_from + v_to)
    q = np.array([np.dot(v_from, v_mid), *np.cross(v_from, v_mid)])
    return q

def normalized(v):
    norm = np.linalg.norm(v)
    return v / norm

NO_STATES = 13
IDX_POS_X = 0
IDX_POS_Y = 1
IDX_POS_Z = 2
IDX_VEL_X = 3
IDX_VEL_Y = 4
IDX_VEL_Z = 5
IDX_QUAT_W = 6
IDX_QUAT_X = 7
IDX_QUAT_Y = 8
IDX_QUAT_Z = 9
IDX_OMEGA_X = 10
IDX_OMEGA_Y = 11
IDX_OMEGA_Z = 12

class Robot:
    
    '''
    frames:
        B - body frame
        I - inertial frame
    states:
        p_I - position of the robot in the inertial frame (state[0], state[1], state[2])
        v_I - velocity of the robot in the inertial frame (state[3], state[4], state[5])
        q - orientation of the robot (w=state[6], x=state[7], y=state[8], z=state[9])
        omega - angular velocity of the robot (state[10], state[11], state[12])
    inputs:
        omega_1, omega_2, omega_3, omega_4 - angular velocities of the motors
    '''
    def __init__(self):
        self.m = 1.0 # mass of the robot
        self.arm_length = 0.25 # length of the quadcopter arm (motor to center)
        self.height = 0.05 # height of the quadcopter
        self.body_frame = np.array([(self.arm_length, 0, 0, 1),
                                    (0, self.arm_length, 0, 1),
                                    (-self.arm_length, 0, 0, 1),
                                    (0, -self.arm_length, 0, 1),
                                    (0, 0, 0, 1),
                                    (0, 0, self.height, 1)])

        self.J = 0.025 * np.eye(3) # [kg m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.constant_thrust = 10e-4
        self.constant_drag = 10e-6
        self.omega_motors = np.array([0.0, 0.0, 0.0, 0.0])
        self.state = self.reset_state_and_input(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        self.time = 0.0
        self.a_adapt = np.zeros(3) # this will need to be changed to 9


        # to log the info:
        self.p_d_I = np.array([0.0, 0.0, 0.0])
        self.v_d_I = np.array([0.0, 0.0, 0.0])
        self.fa = np.array([0.0, 0.0, 0.0])
        self.R_B_to_I = np.eye(3)
        self.T_sp = np.array([0.0, 0.0, 0.0, 0.0])
        self.q_sp = np.array([0.0, 0.0, 0.0, 0.0])
        # self.pi = # initialize diag to 0.1


        self.phi = None # phi from NF load 


    def reset_state_and_input(self, init_xyz, init_quat_wxyz):
        state0 = np.zeros(NO_STATES)
        state0[IDX_POS_X:IDX_POS_Z+1] = init_xyz
        state0[IDX_VEL_X:IDX_VEL_Z+1] = np.array([0.0, 0.0, 0.0])
        state0[IDX_QUAT_W:IDX_QUAT_Z+1] = init_quat_wxyz
        state0[IDX_OMEGA_X:IDX_OMEGA_Z+1] = np.array([0.0, 0.0, 0.0])
        return state0

    # def LOG_DATA(self, windAcc, p_d_I, v_d_I, omega_motor, tau, thrust):
    def LOG_DATA(self):
        t = self.time
        p = self.state[IDX_POS_X:IDX_POS_Z+1]
        p_str = str(p.tolist())

        p_d_I = self.p_d_I
        p_d_I_str = str(p_d_I.tolist())  # assuming p_d_I is array-like

        v = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        v_str = str(v.tolist())

        v_d_I = self.v_d_I
        v_d_I_str = str(v_d_I.tolist())

        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        q_str = str(q.tolist())

        # Rotation matrix.
        R_B_to_I = self.R_B_to_I
        if isinstance(R_B_to_I, (list, np.ndarray)):
            R_B_to_I_str = str(np.array(R_B_to_I).tolist())
        else:
            R_B_to_I_str = str(R_B_to_I)

        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        omega_str = str(omega.tolist())

        # For scalar values, wrap them in a list so they appear with [].
        T_sp = self.T_sp
        T_sp_str = str([T_sp])

        q_sp = self.q_sp
        q_sp_str = str(q_sp.tolist())  # assuming q_sp is array-like

        hover_throttle = 0.5
        hover_throttle_str = str([hover_throttle])
        # v = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        # q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        # w = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        # R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        # omega = omega_motor  # Normalize PWM values
        # fa = windAcc

        fa = self.fa
        # If fa is a NumPy array, convert it to list; otherwise, wrap in list.
        if isinstance(fa, np.ndarray):
            fa_str = str(fa.tolist())
        else:
            fa_str = str([fa])

        # Compute pwm as a NumPy array and convert to list-string.
        pwm = self.omega_motors # Techinically not pwm but its fine for now.
        pwm_str = str(pwm.tolist())

        with open(LOG_FILE6, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([t, p_str, p_d_I_str, v_str, v_d_I_str, q_str,
                R_B_to_I_str, omega_str, T_sp_str, q_sp_str,
                hover_throttle_str, fa_str, pwm_str])
    



    def update(self, omegas_motor, dt, wind_accel=np.zeros(3)):
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]
        R = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        thrust = self.constant_thrust * np.sum(omegas_motor**2)
        f_b = np.array([0, 0, thrust])
        
        tau_x = self.constant_thrust * (omegas_motor[3]**2 - omegas_motor[1]**2) * 2 * self.arm_length
        tau_y = self.constant_thrust * (omegas_motor[2]**2 - omegas_motor[0]**2) * 2 * self.arm_length
        tau_z = self.constant_drag * (omegas_motor[0]**2 - omegas_motor[1]**2 + omegas_motor[2]**2 - omegas_motor[3]**2)
        tau_b = np.array([tau_x, tau_y, tau_z])

        v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81]) #wind here is the wind acceleration
        #  v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81]) + F_wind
        omega_dot = self.J_inv @ (np.cross(self.J @ omega, omega) + tau_b)
        q_dot = 1 / 2 * quat_mult(q, [0, *omega])
        p_dot = v_I 
        
        x_dot = np.concatenate([p_dot, v_dot, q_dot, omega_dot])
        self.state += x_dot * dt
        self.state[IDX_QUAT_W:IDX_QUAT_Z+1] /= np.linalg.norm(self.state[IDX_QUAT_W:IDX_QUAT_Z+1]) # Re-normalize quaternion.
        self.time += dt

        # if self.time == 10:
        #     return -1

    def control(self, p_d_I, v_d_I = None, windAcc = None):
        p_I = self.state[IDX_POS_X:IDX_POS_Z+1]
        v_I = self.state[IDX_VEL_X:IDX_VEL_Z+1]
        q = self.state[IDX_QUAT_W:IDX_QUAT_Z+1]
        omega_b = self.state[IDX_OMEGA_X:IDX_OMEGA_Z+1]

        self.p_d_I = p_d_I
        self.v_d_I = v_d_I

        R_B_to_I = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        self.R_B_to_I = R_B_to_I
        R_I_to_B = R_B_to_I.T

        # Position controller.
        # simple PD right now 
        # Position controller.
        k_p = 1.0
        k_d = 10.0
        v_r = - k_p * (p_I - p_d_I)

        # g_I  = np.array([0, 0, 9.81])
        # f = self.m * (g_I) 

        a = -k_d * (v_I - v_r) + np.array([0, 0, 9.81]) + windAcc
        # a = -k_d * (v_I - v_r) 
        # u_pd = self.m * a # PD controller
        f_wind = self.m * windAcc
        # f += f_wind

        

        f = self.m * a
        self.fa = f_wind 
        f_b = R_I_to_B @ f
        thrust = np.max([0, f_b[2]])
        self.T_sp = thrust

        # f_b = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ f
        # thrust = np.max([0, f_b[2]])

        
        # Attitude controller.
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
        self.q_sp = q_ref
        q_err = quat_mult(quat_conjugate(q_ref), q) # error from Body to Reference.
        if q_err[0] < 0:
            q_err = -q_err
        k_q = 20.0
        k_omega = 100.0
        omega_ref = - k_q * 2 * q_err[1:]
        alpha = - k_omega * (omega_b - omega_ref)
        tau = self.J @ alpha + np.cross(omega_b, self.J @ omega_b)
        
        # Compute the motor speeds.
        B = np.array([
            [self.constant_thrust, self.constant_thrust, self.constant_thrust, self.constant_thrust],
            [0, -self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust],
            [-self.arm_length * self.constant_thrust, 0, self.arm_length * self.constant_thrust, 0],
            [self.constant_drag, -self.constant_drag, self.constant_drag, -self.constant_drag]
        ])
        B_inv = np.linalg.inv(B)
        omega_motor_square = B_inv @ np.concatenate([np.array([thrust]), tau])
        omega_motor = np.sqrt(np.clip(omega_motor_square, 0, None))
        self.omega_motors = omega_motor 
        # self.LOG_DATA(windAcc, p_d_I, v_d_I, omega_motor, tau, thrust)
        return omega_motor

PLAYBACK_SPEED = 1
CONTROL_FREQUENCY = 200 # Hz for attitude control loop
dt = 1.0 / CONTROL_FREQUENCY
time = [0.0]

def get_pos_full_quadcopter(quad):
    """ position returns a 3 x 6 matrix 
        where row is [x, y, z] column is m1 m2 m3 m4 origin h
    """
    origin = quad.state[IDX_POS_X:IDX_POS_Z+1]
    quat = quad.state[IDX_QUAT_W:IDX_QUAT_Z+1]
    rot = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()
    wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
    quadBodyFrame = quad.body_frame.T
    quadWorldFrame = wHb.dot(quadBodyFrame)
    pos_full_quad = quadWorldFrame[0:3]
    return pos_full_quad

def control_propellers(quad):
    t = quad.time
    T = 1.5
    r = 2*np.pi * t / T

    wind_accel = np.zeros(3) #initialize wind accel
    print(t)
    if t > 2:
        # for now, keep the wind constant and only on one direction
        # wind_accel[0] = 10.0
        wind_accel[0] = 30.0  #

    # giving it the figure 8 path
    # FIXME: give it different traj for data collection
    # for figure 8:
    # p_d_I = np.array([np.cos(r/2), np.sin(r), 0.0])
    # v_d_I = np.array([-np.pi/T * np.sin(r/2), 2*np.pi/T * np.cos(r), 0.0])
    # prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel) # figure 8

    # # for a circle:
    p_d_I = np.array([np.cos(r), np.sin(r), 0.0])
    v_d_I = np.array([-2*np.pi/T * np.sin(r), 2*np.pi/T * np.cos(r), 0.0])
    prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel)


    #for a spiral circle:
    # p_d_I = np.array([np.cos(r), np.sin(r),  2*np.sin(r/2) ])
    # v_d_I = np.array([-2*np.pi/T * np.sin(r), 2*np.pi/T * np.cos(r), 2*np.pi/T * np.cos(r/2)])
    # prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel)

    # Note: for Hover mode, just replace the desired trajectory with [1, 0, 1]

    # prop_thrusts = np.array([0, 0, 0, 50])
    #for hover:
    # p_d_I = np.array([0, 0, 0])
    # v_d_I = np.array([0, 0, 0])
    # prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel) 

    quad.update(prop_thrusts, dt, wind_accel)
    

def main():
    quadcopter = Robot()
    def control_loop(i):
        for _ in range(PLAYBACK_SPEED):
            control_propellers(quadcopter)
        quadcopter.LOG_DATA()
        return get_pos_full_quadcopter(quadcopter)

    plotter = QuadPlotter()
    plotter.plot_animation(control_loop)

if __name__ == "__main__":
    main()
