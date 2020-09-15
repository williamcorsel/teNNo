from internal import drone as dr
from internal import player as pl

from internal import keyboard_controller as kc
from internal import waypoint_controller as wpc
from internal import waypoint as wp
from internal import position as pos
from internal import avoidance as ba

import traceback
import logging
import sys
import argparse
import getopt
import os
from datetime import datetime

logging.basicConfig(format="%(name)s: %(levelname)s: %(message)s" ,level=logging.INFO)

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Tello controller with obstacle avoidance.")
    parser.add_argument('--test', '-t', action='store', dest='TESTMODE', choices=['ratio_size', 'ratio_kp', 'avoid'])
    parser.add_argument('--detector', "-d", action="store", dest="OPDETECTOR", choices=["sift", "tf", "yolodark", "eff"])
    parser.add_argument('--zfile', '-z', action='store', dest="ZONEFILE", default="./internal/detectors/yolodark/data/coco.zones")
    parser.add_argument('--output', '-o', action='store', dest="OUTPUTFILE")
    parser.add_argument('--record', '-r', action='store_true', dest="RECORD", default=False)
    args = parser.parse_args()

    if args.ZONEFILE is not None and not os.path.isfile(args.ZONEFILE):
        print("File " + str(args.ZONEFILE) + " is not found.")
        exit(1)

    logfile = args.OUTPUTFILE

    if args.TESTMODE == 'ratio_size':
        if logfile is None:
            logfile = "./logs/ratio_log_"+ datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"
    elif args.TESTMODE == 'ratio_kp':
        if logfile is None:
            logfile = "./logs/ratio_log_" + datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"
    elif args.TESTMODE == 'avoid':
        if logfile is None:
            logfile = "./logs/avoid_log_" + datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".txt"


    drone = dr.Drone()
    avoider = ba.Avoidance(drone, logfile)
    
    # Select detector
    if args.OPDETECTOR == "sift":
        from internal.detectors.sift import sift_detection as d
        detector = d.Sift_Detection(avoider, args.TESTMODE, logfile)
    elif args.OPDETECTOR == "tf":
        from internal.detectors.tf import tf_detection as d
        detector = d.Tf_Detection(avoider)
    elif args.OPDETECTOR == "yolodark":
        from internal.detectors.yolodark import yolo_dark as d
        detector = d.Yolo_Dark_Detection(avoider, args.ZONEFILE, args.TESTMODE, logfile)
    elif args.OPDETECTOR == "eff":
        from internal.detectors.efficientdet import efficientdet_detection as d 
        detector = d.Efficientdet_Detection(avoider)
    else:
        from internal.detectors import base_detection as d
        detector = d.Base_Detection(None, None, logfile=logfile)
    
    player = pl.Player(drone, detector, args.RECORD)
    controller = kc.Keyboard_Controller(drone, player, detector, args.TESTMODE, args.RECORD)
    
    try:
        drone.connect()
        drone.log_data()
        controller.init_controls()
        drone.init_video()
        player.show_video()
        
    except Exception:
        print(traceback.format_exc())
    finally:
        #player.show_possible_obstacle()
        drone.tello.land()
        drone.tello.quit()

if __name__ == '__main__':
    main()
