import rps.robotarium as robotarium
import rps.utilities.graph as graph
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time
import os
import argparse

def move_to_goal(r, N, si_barrier_cert, goal_points, goal_velocities, customUpdate = True):

    # if customUpdate is disabled then 'robotarium' default move to goal executes

    x = r.get_poses()

    if customUpdate:

        for _ in range(9):
            # just use the velocities from the neural network or the file
            x = r.get_poses()
            r.set_velocities(np.arange(N), single_integrator_to_unicycle2(goal_velocities, x))
            r.step()
            # total runtime of the for loop is matched to 0.3 sec approximately
            time.sleep(0.033)

    else:
        # While the number of robots at the required poses is less
        # than N...
        while (np.size(at_pose(x, goal_points, rotation_error=100)) != N):

            # Get poses of agents
            x = r.get_poses()
            x_si = x[:2, :]

            # Create single-integrator control inputs
            dxi = single_integrator_position_controller(x_si, goal_points[:2, :], magnitude_limit=0.08)

            # Create safe control inputs (i.e., no collisions)
            dxi = si_barrier_cert(dxi, x_si)

            # Set the velocities by mapping the single-integrator inputs to unciycle inputs
            #r.set_velocities(np.arange(N), single_integrator_to_unicycle2(dxi, x))
            r.set_velocities(np.arange(N), single_integrator_to_unicycle2(dxi, x))
            # Iterate the simulation
            r.step()



def get_orientation(velocities):
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    cos_sin = velocities / speeds
    orientations = np.arccos(cos_sin[:, :1, :]) * np.sign(cos_sin[:, 1:, :])
    return np.nan_to_num(orientations)


def main():

    # Load trajectory points
    trajectories = np.loadtxt("chaser_agents_20_not_random_pos_vel.txt",delimiter=',')
    # trajectories = np.loadtxt("text_data/c1/agents.txt")

    # Instantiate Robotarium object
    N = int(trajectories.shape[1]/4)
    r = robotarium.Robotarium(number_of_agents=N, show_figure=True, save_data=True, update_time=0.3)

    # Define goal points by removing orientation from poses
    goal_points = generate_initial_conditions(N)
    # Create barrier certificates to avoid collision
    si_barrier_cert = create_single_integrator_barrier_certificate(N)

    positions = np.take(trajectories, [range(0, N *4, 4), range(1, N*4, 4)], axis=1)
    velocities = np.take(trajectories, [range(2, N*4, 4), range(3, N*4, 4)], axis=1)

    # update the orientation from the velocities
    orientations = get_orientation(velocities)

    goal_states = np.concatenate([positions, orientations], axis=1)

    move_to_goal(r, N, si_barrier_cert, goal_states[0,:,:], velocities[0,:,:], False)

    r.call_at_scripts_end()


    time.sleep(0.3)

    print(f'error = {np.linalg.norm(r.get_poses()[:2,:] - positions[0,:,:])}')

    for i in range(goal_states.shape[0]-1):
        print(f'Step {i+2}')
        goal_points = goal_states[i+1, :, :] 
        goal_velocities = velocities[i+1,:,:]

        print(r.previous_render_time)
        move_to_goal(r, N, si_barrier_cert, goal_points, goal_velocities)
        print(r.previous_render_time)

        print(f'error = {np.linalg.norm(r.get_poses()[:2,:] - positions[i+1,:,:])}')

        # Always call this function at the end of your scripts!  It will accelerate the
        # execution of your experiment
        r.call_at_scripts_end()

    time.sleep(5)

if __name__ == '__main__':

    main()
