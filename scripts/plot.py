'''
Plot script used to generate 3d and 2d plot of flight paths.
Supports a maximum of 10 plots
'''

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import argparse


color_list = [
    "b",
    "g",
    "r",
    "c",
    "m",
    "y",
    "cyan",
    "orange",
    "purple",
    "saddlebrown"
]

def main():
    parser = argparse.ArgumentParser(description="Plotting program.")
    parser.add_argument("--plot", "-p", dest="PLOT", choices=["3d", "2d"])
    parser.add_argument("--input", "-i", nargs="+", dest="INPUT")
    parser.add_argument("--output", "-o", dest="OUTPUT", default="output.txt")
    args = parser.parse_args()

    plot_list = read_plots(args.INPUT)

    if args.PLOT == "3d":
        plot3d(plot_list)
    elif args.PLOT == "2d":
        plot2d(plot_list)

def read_plots(input):
    plot_list = []

    for file in input:
        x,y,z = np.loadtxt(file, delimiter=',', unpack=True, skiprows=1)
        plot_list.append([x,y,z])

    return plot_list   

def plot2d(plot_list):
    fig = plt.figure(figsize=(10,7))
    ax = plt.axes()

    ax.set_xlabel("X-axis (m)")
    ax.set_ylabel("Y-axis (m)")
    ax.invert_xaxis()

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    ax.set_xlim([3,0])
    ax.set_ylim([2,-2])
    ax.text(0.15,-0.2, "start")
    
    # Obstacle
    rect = patches.Rectangle((2,-0.25), 0.2, 0.5, facecolor="black", edgecolor="black", zorder=10)
    ax.add_patch(rect)

    for i in range(0, len(plot_list)):
        plot = plot_list[i]
        ax.plot(plot[0], plot[1], color=color_list[i])

    
    plt.show()


def plot3d(plot_list):
    
    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection="3d")  
    
    #Obstacle
    ax.bar3d(2, -0.25, -1, 0.2, 0.5, 1.8, color="black")
    
    for i in range(0, len(plot_list)):
        plot = plot_list[i]
        ax.plot(plot[0], plot[1], plot[2], color=color_list[i], zorder=10)

    ax.set_xlabel("X-axis (m)")
    ax.set_ylabel("Y-axis (m)")
    ax.set_zlabel("Z-axis (m)")
    ax.set_xlim([0,3])
    ax.set_ylim([2,-2])
    ax.set_zlim([-1,2])

    ax.text(0,0.3,0, "start")
    ax.view_init(30, 120)
    
    plt.show()


if __name__ == '__main__':
    main()