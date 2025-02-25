import scipy.spatial.transform
import numpy as np
from animate_function import QuadPlotter
import sys
sys.path.append('/home/anahita/CDS245/HW2_new/neural-fly')
import mlmodel
# from mlmodel import Phi_Net, load_model
import torch 
import csv

def build_block_matrix_Phi(phi_vec):
    phi = np.array(phi_vec).reshape(1, 3)
    return np.kron(np.eye(3), phi)

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

        self.p_d_I = np.array([0.0, 0.0, 0.0])
        self.v_d_I = np.array([0.0, 0.0, 0.0])
        self.fa = np.array([0.0, 0.0, 0.0])
        self.R_B_to_I = np.eye(3)
        self.T_sp = np.array([0.0, 0.0, 0.0, 0.0])
        self.q_sp = np.array([0.0, 0.0, 0.0, 0.0])


        # for NF
        self.a_adapt = np.zeros((9,1)) # this will need to be changed to 9
        self.p = (0.001)* np.eye(9)
        self.lamb = 0.01
        self.R = 100 * np.eye(3)
        self.Q = (1e-4)*np.eye(9)
    

        # self.phi = Phi_Net()
        # self.phi = load_model("simulation_dim-a-3_v-q-pwm-epoch-800",   "/home/anahita/CDS245/HW2_new/neural-fly/models/").phi
        # Low Pass Phi
        self.smooth_Phi = None
        
    
    # #######################################################################

    def reset_state_and_input(self, init_xyz, init_quat_wxyz):
        state0 = np.zeros(NO_STATES)
        state0[IDX_POS_X:IDX_POS_Z+1] = init_xyz
        state0[IDX_VEL_X:IDX_VEL_Z+1] = np.array([0.0, 0.0, 0.0])
        state0[IDX_QUAT_W:IDX_QUAT_Z+1] = init_quat_wxyz
        state0[IDX_OMEGA_X:IDX_OMEGA_Z+1] = np.array([0.0, 0.0, 0.0])
        return state0

    # #######################################################################
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

        # v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81]) + wind_accel #wind here is the wind acceleration
        v_dot = 1 / self.m * R @ f_b + np.array([0, 0, -9.81])  #wind here is the wind acceleration
        omega_dot = self.J_inv @ (np.cross(self.J @ omega, omega) + tau_b)
        q_dot = 1 / 2 * quat_mult(q, [0, *omega])
        p_dot = v_I 
        
        x_dot = np.concatenate([p_dot, v_dot, q_dot, omega_dot])
        self.state += x_dot * dt
        self.state[IDX_QUAT_W:IDX_QUAT_Z+1] /= np.linalg.norm(self.state[IDX_QUAT_W:IDX_QUAT_Z+1]) # Re-normalize quaternion.
        self.time += dt



    def low_pass_filter(self, Phi):
        if self.smooth_Phi is None:
            self.smooth_Phi = Phi
        else:
            alpha = 0.3
            self.smooth_Phi = alpha * Phi + (1 - alpha) * self.smooth_Phi
        return self.smooth_Phi


    # ############################################################################################
    def control(self, p_d_I, v_d_I = None, windAcc = None, model=None):
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
        # FIXME
        k_p = 1.0
        k_d = 10.0
        k_i = 1
        fa = self.m * windAcc
        f = fa

        
        v_r = - k_p * (p_I - p_d_I)

        # v_r = v_d_I - k_p * (p_I - p_d_I)
        s = (v_I - v_r).reshape(3, 1) 
        a = -k_d * (v_I - v_r) + np.array([0, 0, 9.81]) 

        u_pd = self.m * a # PD controller

        # f = self.m * (np.array([0, 0, 9.81]) + fa)
        # f =  u_pd

        tensor_input = torch.from_numpy(np.concatenate([v_I, q, self.omega_motors])).flatten()
        phi_val  = model.phi(tensor_input).detach().numpy().reshape(-1)
        phi_val = np.array(phi_val).reshape(3,1)
        phi_notFilter  = build_block_matrix_Phi(phi_val) # 3 x 9

        # with torch.no_grad():
        # phi_net =  self.phi(tensor_input).detach().numpy().reshape(-1)
        # phi_net = np.array(phi_net).reshape(3,1)
        # phi_notFilter  = build_block_matrix_Phi(phi_net) # 3 x 9
        # phi_net =  self.phi(tensor_input).numpy()
        # zeros = np.zeros_like(phi_net)
        # phi_notFilter = np.vstack([np.concatenate([phi_net, zeros, zeros]),
        #                                 np.concatenate([zeros, phi_net, zeros]),
        #                                 np.concatenate([zeros, zeros, phi_net])])

        phi = self.low_pass_filter(phi_notFilter)
        phi_T = np.transpose(phi)
        # this adaptation comes after training 
        # put the origoinal controller back for data collection
        # after data collection, for NF 
        # a = -K * s + np.array([0, 0, 9.81]) - (phi@self.a_adapt)
        # self.a_adapt = self.a_adapt +   (-lamb* self.a_adapt + self.p @ phi.T @s)*dt
        u_NF= (-phi @ self.a_adapt).flatten()
        a_adapt_dot = -self.lamb * self.a_adapt - (self.p @ phi_T @ np.linalg.inv(self.R)) @ (phi @ self.a_adapt - fa.reshape(3,1) ) + self.p @ phi_T @ s
        self.a_adapt = self.a_adapt +  (a_adapt_dot)*dt

        p_dot = (-2)* self.lamb  * self.p + self.Q - self.p @ phi_T @  np.linalg.inv(self.R) @ phi @ self.p
        self.p = self.p + (p_dot)*dt
        # max_val = 40


        # u_NF= (-phi @ self.a_adapt).flatten()

        # a_min = np.array([-max_val, -max_val, -max_val])  # Lower bound
        # a_max = np.array([max_val, max_val, max_val])  # Upper bound
        
        # a = np.clip(a, a_min, a_max)
        u = np.clip(u_pd + u_NF, -30, 30)
        #u = np.clip(u_pd, -15, 15)
        f += u
        # print(a)
        # print("Blowing Wind: ", fa)
        print("u_nf ", (- phi @ self.a_adapt).flatten())
        print("|error|  ",np.linalg.norm(phi @ self.a_adapt - fa.reshape(3,1)))
        # # print("|a_dot_pred|  ",np.linalg.norm(a_dot_pred))
        # # print("|a_dot_reg|  ",np.linalg.norm(a_dot_reg))
        # print("|a_dot_track_err|  ",np.linalg.norm( self.p @ phi_T @ s))
        # print("|P|  ",np.linalg.norm(self.p))
        # print("|a|  ",np.linalg.norm(self.a_adapt))
        # print("|Phi|  ",np.linalg.norm(phi))
        # print("Phi: ", phi)
        # print("u: ", u)
        # print("|s|  ",np.linalg.norm(s))
        # print("-----------------------------")
        
        # a_adapt = a_adapt + ...

        # f = self.m * a
        # f_b = scipy.spatial.transform.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ f
        # thrust = np.max([0, f_b[2]])
        self.fa = fa 
        f_b = R_I_to_B @ f
        thrust = np.max([0, f_b[2]])
        self.T_sp = thrust
        # Attitude controller.
        q_ref = quaternion_from_vectors(np.array([0, 0, 1]), normalized(f))
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
        self.omega_motors = np.sqrt(np.clip(omega_motor_square, 0, None))
        return self.omega_motors
    
# #######################################################################

PLAYBACK_SPEED = 1
CONTROL_FREQUENCY = 200 # Hz for attitude control loop
dt = 1.0 / CONTROL_FREQUENCY
time = [0.0]

# #######################################################################
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

# #######################################################################
def control_propellers(quad, model):
    t = quad.time
    T = 1.5
    r = 2*np.pi * t / T

    wind_accel = np.zeros(3) #initialize wind accel
    print(t)
    if t > 2:
        # for now, keep the wind constant and only on one direction
        wind_accel[0] = 5.0  #

    # giving it the figure 8 path
    # FIXME: give it different traj for data collection
    # for figure 8:
    # p_d_I = np.array([np.cos(r/2), np.sin(r), 0.0])
    # v_d_I = np.array([-np.pi/T * np.sin(r/2), 2*np.pi/T * np.cos(r), 0.0])
    # prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel, model=model) # figure 8

    # # for a circle:
    p_d_I = np.array([np.cos(r), np.sin(r), 0.0])
    v_d_I = np.array([-2*np.pi/T * np.sin(r), 2*np.pi/T * np.cos(r), 0.0])
    prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel, model=model)


    #for a spiral circle:
    # p_d_I = np.array([np.cos(r), np.sin(r),  2*np.sin(r/2) ])
    # v_d_I = np.array([-2*np.pi/T * np.sin(r), 2*np.pi/T * np.cos(r), 2*np.pi/T * np.cos(r/2)])
    # prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel, model=model)

    # Note: for Hover mode, just replace the desired trajectory with [1, 0, 1]

    # prop_thrusts = np.array([0, 0, 0, 50])
    #for hover:
    # p_d_I = np.array([0, 0, 0])
    # v_d_I = np.array([0, 0, 0])
    # prop_thrusts = quad.control(p_d_I, v_d_I, wind_accel, model=model) 

    quad.update(prop_thrusts, dt, wind_accel)

# #######################################################################
def main():
    #Load in model 
    dim_a = 3
    features = ['v', 'q', 'pwm']
    dataset = 'simulation' 

    modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}"
    stopping_epoch = 200
    model = mlmodel.load_model(modelname = modelname + '-epoch-' + str(stopping_epoch), modelfolder = "/home/anahita/CDS245/HW2_new/neural-fly/models/")
# _dim-a-3_v-q-pwm-epoch-800",   "/home/anahita/CDS245/HW2_new/neural-fly/models/").phi
    quadcopter = Robot()
    def control_loop(i):
        for _ in range(PLAYBACK_SPEED):
            control_propellers(quadcopter, model)
        return get_pos_full_quadcopter(quadcopter)

    plotter = QuadPlotter()
    plotter.plot_animation(control_loop)

if __name__ == "__main__":
    main()



        # phi_net = self.phi(v_I, q,  )
        # print("phi_net ",  phi_net.shape) 
        # print(" phi ", phi.shape)
        # phi = np.diag([phi_net, phi_net, phi_net])

        
        
        # pi = somethingh
        # v_d_I = np.zeros(3)