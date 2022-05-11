#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish.

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # DONE: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Waypoint)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # DONE: Add other member variables you need below
        self.pose = None
        self.base_lane = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None
        
        self.loop() # Taking control over the publishing frequency with self-implemented loop()

    def loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            # meaning: if given a vehicle pose and a base set of waypoints to operate from
            if self.pose and self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        # Get vehicle coordinates
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        # Query the KDTree for closest point in tree -> gets us closest coordinate's index
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
       
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx + 1]
        
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        
        # Calculate dot product on distance between closest and previous
        # and current and closest vectors
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        
        # If this dot product is positive: closest waypoint is behind the vehicle
        # So we take the next waypoint from that (closest_idx + 1)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx
    
    def publish_waypoints(self):
        lane = self.generate_lane()
        self.final_waypoints_pub.publish(lane)
    
    def generate_lane(self):
        lane = Lane() # Create lane for publishing

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        
        return lane
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = [] # (additional) deceleration waypoint list
        
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            
            # four waypoints back from line so front of car stops at line
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(dist)
            if vel < 1.0:
                vel = 0.0
            
            # We don't want to go higher than the speeds attributed to the original waypoints
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp
    
    def pose_cb(self, msg):
        # DONE: Implement
        self.pose = msg # Store the car's pose

    def waypoints_cb(self, waypoints):
        # DONE: Implement
        # Executed only once, allows for direct copying of entire waypoint set
        self.base_lane = waypoints
        # Making sure waypoints_2d is actually initialized here
        if not self.waypoints_2d:
            # Converting waypoints to two-dimensional points for KDTree
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # Looking up closest waypoint (log n)
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # DONE: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    
if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
