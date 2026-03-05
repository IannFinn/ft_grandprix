from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sys import platform
import os
import re
import inspect
from PIL import Image
import tempfile
import shutil
import math
import importlib
import queue
import json
from dataclasses import dataclass, field
from .vehicle import VehicleStateSnapshot
import mujoco
import mujoco.viewer
import numpy as np
import time
from .colors import colors, resolve_color
import dearpygui.dearpygui as dpg
from .lobotomy import Driver as LobotomyDriver
from threading import Lock, Event, Thread
from .curve import extract_path_from_svg
from .chunk import chunk
from .map import produce_mjcf
from .vendor import Renderer
from .raycast import fakelidar
from .bracket import compute_driver_files
import tracemalloc
import linecache
import types

def tag():
    return field(default_factory=dpg.generate_uuid)

# monkey patch so that the viewer doesn't do its own forwarding
mujoco._mj_forward = mujoco.mj_forward
mujoco.mj_forward = lambda model, data: None

np.set_printoptions(precision=2, formatter={"float": lambda x: f"{x:8.2f}"})

def invert(d): return { v : k for k, v in d.items() }

def readable_keycode(keycode):
    try:
        return chr(keycode).encode("ascii").decode()
    except:
        return keycode

def ordinal(n):
    n = str(n)
    if n == '0' or len(n) > 1 and n[-2] == '1': e = 'th'
    elif (n[-1] == '1'): e = 'st'
    elif (n[-1] == '2'): e = 'nd'
    elif (n[-1] == '3'): e = 'rd'
    else:                e = 'th'
    return n + e

def runtime_import(path):
    spec = importlib.util.find_spec(path)
    module =  spec.loader.load_module()
    return module

def quaternion_to_euler(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return [yaw_z, pitch_y, roll_x]

def quaternion_to_angle(w, x, y, z):
    return quaternion_to_euler(w, x, y, z)[0]

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    return [qw, qx, qy, qz]

exit_event = Event()

class VehicleState:
    def __init__(self, /, id, offset, driver, label, driver_path, data, rangefinders):
        # lifetime of race < lifetime of mujoco, ∴ mujoco is reloaded
        # more often than it needs to be.
        self.id = id
        self.offset = offset
        self.completion = 0 # current completion percentage
        self.good_start = True # if we enter into a lap backwards, set this to False
        self.driver = driver
        self.driver_path = driver_path
        self.finished = False
        self.delta = 0
        self.v2 = len(inspect.signature(self.driver.process_lidar).parameters) >= 2

        self.speed = 0.0
        """the last speed command"""

        self.steering_angle = 0.0
        """the last steering angle command"""

        self.distance_from_track = 0.0
        """distance from the centerline"""

        self.label = label
        """the name of this driver"""
        
        self.start = 0
        """physics time step we are currently at"""

        self.laps = 0
        """number of laps completed"""

        self.times = []
        """lap times so far"""
        self.progress = 0
        """previous compleition - current completion"""

        # lifetime of mujoco
        self.forward     = data.actuator(f"forward #{self.id}").id
        self.turn        = data.actuator(f"turn #{self.id}").id
        self.joint       = data.joint(f"car #{self.id}")
        self.sensors     = [data.sensor(f"rangefinder #{self.id}.#{j}").id for j in range(rangefinders)]

    def lap_completion(self):
        """
        Gets our completion around the track. Returns a negative number if
        we are going backwards.
        """
        if self.good_start:
            return self.completion
        else:
            return -(100 - self.completion)
    
    def absolute_completion(self):
        return self.laps * 100 + self.lap_completion()

    def reload_code(self):
        self.driver = runtime_import(self.driver_path).Driver()
        self.v2 = len(inspect.signature(self.driver.process_lidar).parameters) >= 2

    def snapshot(self, time=0):
        yaw, pitch, roll = quaternion_to_euler(*self.joint.qpos[3:])
        return VehicleStateSnapshot(
            laps = self.laps,
            velocity=self.joint.qvel[:3],
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            lap_completion=self.lap_completion(),
            absolute_completion=self.absolute_completion(),
            time=time
        )

    
# A simple JSON option that can be persisted in the file system
class Option:
    def __init__(self, tag, default, _type=None, callback=None, description=None, data=None, label=None, persist=True, min_value=None, max_value=None, present=True):
        self.tag = tag
        self.dpg_tag = f"__option__::{tag}"
        self.dpg_description_tag = f"__option__::{tag}::__description__"
        self.label = label or tag
        self.persist = persist
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.value = default
        self.present = present
        self.callback = callback
        self.type = _type or type(default)
        if self.persist and self.tag in data:
            their = type(data[self.tag])
            if data[self.tag] is not None and self.type is not their:
                print (f"{tag} was saved as {their}, using default instead ({self.type})")
                self.value = default
            else:
                self.value = data[self.tag]

class Command:
    def __init__(self, tag, callback, label=None, description=None):
        self.tag = tag
        self.label = label or tag
        self.callback = callback
        self.description = description or getattr(callback, "__doc__", None)

class Mujoco:
    def __init__(self, track):
        #print("Running mujoco thread")
        self.reset_event = Event()
        self.hard_reset_event = Event()
        self.running_event = Event()
        self.running_event.set()
        self.launch_viewer_event = Event()
        self.kill_viewer_event = Event()
        self.mv = None
        self.camera_vel = [0, 0]
        self.camera_pos_vel = [0, 0, 0]
        self.camera_friction = [0.01, 0.01]
        self.watching = 0
        self.rendered_dir = "rendered"
        self.template_dir = "template"
        self._camera = mujoco.MjvCamera()
        self.options = {}
        self.cars = []
        self.track = track
        try:
            with open("aigp_settings.json") as data_file:
                data = json.load(data_file)
        except Exception as e:
            data = {}
            print("Not loading settings from file: ", e)
        
        self.declare("sort_vehicle_list", False, label="Sort Vehicles", description="Keeps the vehicle list sorted by position", data=data)
        self.declare("reset_camera", True, label="Reset Camera", data=data,
                     description="Resetting sets sets the camera position ot the first path point")
        self.declare("option_intensity", 1.0, label="Icon Intensity", data=data, callback=self.set_icon_intensity)
        self.declare("lock_camera", False, label="Lock Camera", data=data,
                     description="If this option is set, the camera angle will be kept in line with the angle of the vehicle being watched")
        self.declare("detach_control", False, label="Detached Control", data=data,
                     description="Do not attempt to deliver controls to the car being watched")
        self.declare("manual_control", False, label="Manual Control", persist=False,
                     description="Control the car being watched with the W, A, S and D keys")
        self.declare("always_invoke_driver", True, data=data, label="Invoke with Manual", present=False,
                     description="Will keep on invoking the process LiDAR function even if manual control is set")
        self.declare("manual_control_speed", 3.0, label="Manual Control Speed", data=data,
                     description="The speed at which the car will drive when being controlled manually")
        self.declare("cars_path", "cars.json", _type=str, label="Cars Path", persist=False)
        self.declare("lap_target", 10, data=data, label="Lap Target")
        self.declare("max_fps", 30, data=data, label="Max FPS")
        self.declare("cinematic_camera", False, data=data, label="Cinematic Camera")
        self.declare("center_camera", False, data=data, label="Center Camera")
        self.declare("center_camera_inside", True, data=data, label="Center Camera Inside", present=False,
                     description="If this option is set to true, the 'center_camera' option will view the vehicle being watched from the inside of the track")
        self.declare("pause_on_reload", True, data=data, label="Pause on Reload")
        self.declare("save_on_exit", True, data=data, label="Save on Exit", present=False,
                     description="Save to `aigp_settings.json` on exit")
        self.declare("bubble_wrap", False, label="Soften Collisions", data=data, persist=True,
                     description="Soften collisions with the map border so that vehicles don't get stuck as easilly",
                     callback=self.soften)
        self.declare("physics_fps", 500, data=data, label="Max Physics FPS", present=False,
                     description="The reciprocal of the physics timestep passed into mujoco")
        self.declare("max_geom", 1500, data=data, label="Mujoco Geom Limit", persist=False, present=False,
                     description="The number of entities that mujoco will render")
        self.declare("rangefinder_alpha", 0.1, label="Rangefinder Intensity", data=data, callback=self.rangefinder)
        self.declare("tricycle_mode", False, label="Use a Tricycle 🤡", data=data, present=False, persist=True,
                     description="Use the old differential drive vehicle model (requires hard reset)",
                     callback=lambda x: self.hard_reset_event.set())
        self.declare("naive_flatten", False, label="Naive Flatten", data=data, present=False, persist=True,
                     description="Prevent vehicles from rotating in a naive manner. Violates some constraints of the physics engine and may lead to cars flying away.")
        self.declare("debug_mode", False, label="Debug Mode", data=data,
                     description="Shows hidden debugging settings")
        self.declare("map_color", [1, 0, 0, 1], label="Map Color", data=data, present=False)
        self.declare("rangefinder_tilt", 0.0, label="Rangefinder Tilt", data=data, present=False, min_value=0, max_value=np.pi)
        self.declare("use_simulated_simulation_lidar", False, label="Simulate Lidar", data=data, present=False,
                     description="If this is used, mujoco's slow raycasting will be avoided and raycasting will instead be performed using the 2D image and vehicle size and position metadata (fater)",
                                  callback=self.set_use_simulated_simulation_lidar_flag)

        self.vehicle_states = []
        self.shadows = {}
        self.steps = 0

        self.viewer = None
        self.kill_inline_render_event = Event()
        self.render_finished = Event()
        self.render_finished.set()

    def set_use_simulated_simulation_lidar_flag(self, flag):
        if flag:
            for vehicle_state in self.vehicle_states:
                self.shadow_rangefinders(vehicle_state.id)
        else:
            for vehicle_state in self.vehicle_states:
                if vehicle_state.id not in self.shadows:
                    self.unshadow_rangefinders(vehicle_state.id)
            

    def set_icon_intensity(self, intensity):
        for i in range(len(self.vehicle_states)):
            self.model.mat(f"car #{i} icon").rgba[:4] = intensity

    @property
    def camera(self):
        if self.viewer is None or type(self.viewer) is str:
            return self._camera
        else:
            return self.viewer.cam

    def perturb_camera_pos(self, dx, dy, dz):
        if self.option("cinematic_camera"):
            if self.watching is not None:
                target = self.data.body(f"car #{self.watching}").xpos[:]
                delta = target - self.camera.lookat
                delta_magnitude = np.linalg.norm(delta)
                next_lookat = target - (self.camera.lookat + self.camera_pos_vel)
                next_lookat_magnitude = np.linalg.norm(next_lookat)
                P = 0.1 * delta
                D = 0.05 * (next_lookat_magnitude - delta_magnitude) * delta
                self.camera_pos_vel = P + D

    def perturb_camera(self, dx, dy):
        if not self.option("cinematic_camera"):
            self.camera.azimuth   += dx
            self.camera.elevation += dy
        else:
            self.camera_vel[0] += dx / 100
            self.camera_vel[1] += dy / 100

    def soften(self, soften=True):
        try:
            if self.mushr:
                for vehicle_state in self.vehicle_states:
                    self.model.geom(f"bl softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"br softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"fr softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"fl softener #{vehicle_state.id}").conaffinity = int(soften) << 2
            else:
                for vehicle_state in self.vehicle_states:
                    self.model.geom(f"left softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"right softener #{vehicle_state.id}").conaffinity = int(soften) << 2
                    self.model.geom(f"front softener #{vehicle_state.id}").conaffinity = int(soften) << 2
        except Exception as e:
            print("Error configuring collision softening geoms.", e)

    def run(self):
        self.stage()
        self.reload()
        
    def persist(self):
        if self.option("save_on_exit"):
            options = {
                option.tag : option.value
                for option in self.options.values()
                if option.persist
            }
            path = tempfile.mktemp()
            with open(path, "w") as file:
                json.dump(options, file)
            shutil.copy(path, "aigp_settings.json")
            os.remove(path)

    def declare(self, tag, default, **kwargs):
        self.options[tag] = Option(tag, default, **kwargs)

    def nuke(self, tag):
        self.options[tag].value = None

    def option(self, tag, value=None):
        option = self.options[tag]
        if value is not None:
            # print(f"`{tag}` = `{value}`")
            option.value = value
            if option.callback is not None:
                option.callback(value)
        return option.value
    
    def reload(self):
        if self.option("pause_on_reload"):
            self.running_event.clear()
        mujoco.mj_resetData(self.model, self.data)
        for m in self.vehicle_states:
            self.unshadow(m.id)
        vehicle_states = []
        for i in range(1):
            path = "drivers.v3"
            #print(f"Loading driver from python module path '{path}'")
            driver = runtime_import(path).Driver()
            vehicle_state = VehicleState(
                id           = i,
                offset       = (i+2) * 2,
                driver_path  = path,
                driver       = driver,
                label        = "carling",
                data         = self.data,
                rangefinders = self.mjcf_metadata["rangefinders"]
            )
            vehicle_states.append(vehicle_state)
        self.vehicle_states = vehicle_states
        if self.watching is not None and self.watching >= len(self.vehicle_states):
            self.watching = None
        self.shadows = {}
        self.steps = 0
        self.winners = {}
        if self.option("reset_camera"):
            self.camera.lookat[:2] = self.path[0]
        self.position_vehicles(self.path)
    def rangefinder(self, value):
        self.model.vis.rgba.rangefinder[3] = value
    
    def stage(self, track=None):
        if track is None:
            if self.track is None:
                raise RuntimeError("stage must be called with a track first")
        else:
            self.track = track
        if self.option("cars_path") is not None:
            cars_path = os.path.join(self.template_dir, "cars", self.option("cars_path"))
            try:
                with open(cars_path) as cars_file:
                    self.cars = [{"driver" : "drivers.v3", "name" : "carling", "primary" : "red", "secondary" : "pink", "icon" : "white.png" } for genome in range(1)]
            except Exception as e:
                print(f"ERROR: Could not read from `{cars_path}`")
                self.cars = []
        
        image_path = os.path.join(self.template_dir, f"{self.track}.png")
        balls = np.array(Image.open(image_path))
        balls[balls != 255] = 0
        balls = 255 - balls
        chunk(image_path, verbose=False, force=True, scale=2.0)
        produce_mjcf(
            template_path=os.path.join(self.template_dir, "mushr.em.xml"),
            rangefinders=90,
            cars=self.cars,
            map_color=self.option("map_color")[:3]
        )
        self.mushr = True

        map_metadata_path = os.path.join(self.rendered_dir, "chunks", "metadata.json")
        with open(map_metadata_path) as map_metadata_file:
            self.map_metadata = json.load(map_metadata_file)
        mjcf_metadata_path = os.path.join(self.rendered_dir, "car.json")
        with open(mjcf_metadata_path) as mjcf_metadata_file:
            self.mjcf_metadata = json.load(mjcf_metadata_file)
        self.model_path = os.path.join(self.rendered_dir, "car.xml")
        self.original_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        self.path = extract_path_from_svg(os.path.join(self.template_dir, f"{self.map_metadata['name']}-path.svg"))
        self.path[:, 0] =   self.path[:, 0] / self.map_metadata['width']  * self.map_metadata['chunk_width'] * self.map_metadata['scale']
        self.path[:, 1] = - self.path[:, 1] / self.map_metadata['height'] * self.map_metadata['chunk_height']  * self.map_metadata['scale']
        if platform == "darwin":
            for vehicle_state in self.vehicle_states:
                id = vehicle_state.id
                self.model.light(f"top light #{id}").active = 0
                self.model.light(f"front light #{id}").active = 0
        self.sync_options()

    def sync_options(self):
        """
        Invokes any callbacks associated with options and ensure that the application state
        is valid this can be considered generally safe as options are ready to be set to
        any value at any time in a thread-safe manner.
        """
        for option in self.options.values():
            if option.callback is not None:
                option.callback(option.value)


    def reload_code(self, index):
        self.vehicle_states[index].reload_code()

    def position_vehicles(self, path=None):
        """
        Sends all vehicles back to their starting positions without changing race
        metadata such as number of laps, lap times .etc.
        """
        if path is None:
            path = self.path
        for i, m in enumerate(self.vehicle_states):
            i = m.offset
            delta = path[i+1]-path[i]
            angle = math.atan2(delta[1], delta[0])
            quat = euler_to_quaternion([angle, 0, 0])
            m.joint.qpos[3:] = np.array(quat)
            m.joint.qpos[:2] = path[i]

    def step(self,action_speed,action_turn):
        vehicle_state = self.vehicle_states[0]
        if self.option("naive_flatten"):
            vehicle_state.joint.qpos[3:] = euler_to_quaternion([quaternion_to_angle(*vehicle_state.joint.qpos[3:]), 0, 0])
        xpos = vehicle_state.joint.qpos[0:2]
        distances = ((self.path - xpos)**2).sum(1)
        closest = distances.argmin()
        vehicle_state.distance_from_track = distances[closest]
        vehicle_state.off_track = vehicle_state.distance_from_track > 1
        if not vehicle_state.off_track:
            completion = (closest - vehicle_state.offset) % 100
            delta = completion - vehicle_state.completion
            vehicle_state.delta = (completion - vehicle_state.completion + 50) % 100 - 50
            
            if abs(delta) > 90:
                lap_time = (self.steps - vehicle_state.start) * self.model.opt.timestep
                if vehicle_state.delta < 0:
                    vehicle_state.laps -= 1
                    vehicle_state.good_start = False
                    if len(vehicle_state.times) != 0:
                        vehicle_state.times.pop()
                elif vehicle_state.delta > 0:
                    if vehicle_state.good_start:
                        # dont add a new lap OR start counting
                        # time for the next lap if we went
                        # backwards, we are still in the current
                        # lap
                        vehicle_state.times.append(lap_time)
                        vehicle_state.start = self.steps
                    vehicle_state.laps += 1
                    vehicle_state.good_start = True
            if vehicle_state.laps >= self.option("lap_target"):
                if vehicle_state.id not in self.winners:
                    self.winners[vehicle_state.id] = len (self.winners) + 1
                vehicle_state.finished = True
                self.shadow(vehicle_state.id)
            vehicle_state.progress = completion - vehicle_state.completion
            vehicle_state.completion = completion
            

        id = self.model.sensor("car #0 accelerometer").id
        d = self.data.sensordata[id:id+3]
        # print(f"ACCEL: data: {d} accel: {math.sqrt((d**2).sum())}")
        id = self.model.sensor("car #0 gyro").id
        d = self.data.sensordata[id:id+3]
        # print(f"GYRO:  accel: {math.sqrt((d**2).sum())}")
        ranges = self.data.sensordata[vehicle_state.sensors]

        snapshot = vehicle_state.snapshot(time = self.steps / self.model.opt.timestep)
        if vehicle_state.v2: driver_args = [ranges, snapshot]
        else:                driver_args = [ranges]

        speed, steering_angle = 0.0, 0.0
        speed = action_speed
        steering_angle = action_turn
        # try:
            # speed, steering_angle = vehicle_state.driver.process_lidar(*driver_args)
        # except Exception as e:
            # print(f"Error in vehicle `{vehicle_state.label}`: `{e}`")
        
        vehicle_state.speed = speed
        vehicle_state.steering_angle = steering_angle
        
        if not self.option("detach_control"):
            self.data.ctrl[vehicle_state.forward] = vehicle_state.speed
            self.data.ctrl[vehicle_state.turn] = vehicle_state.steering_angle
            
        mujoco.mj_step(self.model, self.data)
        self.steps += 1

        #p_fps = self.option("physics_fps")
        #now = time.time()
        #fps = 1 / (now - last) if now - last > 0 else p_fps
        # if (now - last < 1/p_fps):
            # time.sleep(1/p_fps - (now - last));
        #last = time.time()
        #print(f"{fps} fps")
        return driver_args,vehicle_state

    def shadow_rangefinders(self, i):
        for j in range(self.mjcf_metadata["rangefinders"]):
            self.model.sensor(f"rangefinder #{i}.#{j}").type = mujoco.mjtSensor.mjSENS_USER

    # send car to the shadow realm
    def shadow(self, i):
        if i in self.shadows:
            return self.shadows[i]
        vehicle_state = self.vehicle_states[i]
        # lobotomise car first
        vehicle_state.driver = LobotomyDriver()
        vehicle_state.v2 = False
        vehicle_state.driver_path = "ft_grandprix.lobotomy"
        shadows = []
        self.shadow_rangefinders(i)
        transparent_material_id  = self.model.mat("transparent").id
        shadow_alpha = 0.1
        for mat in [f"car #{i} primary", f"car #{i} secondary", f"car #{i} body", f"car #{i} wheel", f"car #{i} icon"]:
            self.model.mat(mat).rgba[3] = shadow_alpha
        for geom in self.subgeoms(i):
            m, d = self.model.geom(geom), self.data.geom(geom)
            r = [*self.model.mat(m.matid.item()).rgba[:3], shadow_alpha]
            shadow = dict(type=m.type.item(), size=m.size, pos=d.xpos, rgba=r, mat=d.xmat)
            # turn off collisions (only makes sense wrt car.em.xml condim and
            # conaffinity values)
            m.conaffinity = 0
            m.contype = 1 << 1
            m.matid = transparent_material_id
            shadows.append(shadow)
        self.shadows[vehicle_state.id] = shadows
        return self.shadows[vehicle_state.id]

    def subgeoms(self, i):
        if self.mushr:
            return [f"chasis #{i}",
                    f"car #{i} lidar",
                    f"buddy_wheel_fl_throttle #{i}",
                    f"buddy_wheel_fr_throttle #{i}",
                    f"buddy_wheel_bl_throttle #{i}",
                    f"buddy_wheel_br_throttle #{i}"]
        else:
            return [f"chasis #{i}",
                    f"car #{i} lidar",
                    f"front wheel #{i}",
                    f"left wheel #{i}",
                    f"right wheel #{i}"]

    def unshadow_rangefinders(self, i):
        for j in range(90):
            self.model.sensor(f"rangefinder #{i}.#{j}").type = mujoco.mjtSensor.mjSENS_RANGEFINDER

    def unshadow(self, i):
        if i not in self.shadows:
            return
        self.unshadow_rangefinders(i)
        for mat in [f"car #{i} primary", f"car #{i} secondary", f"car #{i} body", f"car #{i} wheel", f"car #{i} icon"]:
            self.model.mat(mat).rgba[3] = 1.0
        for geom in self.subgeoms(i):
            m = self.model.geom(geom)
            m.conaffinity = 1
            m.contype = 1
            m.matid = self.original_model.geom(geom).matid
        del self.shadows[i]


    





def display_top(snapshot, key_type='lineno', limit=100):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def profile():
    print("Profiling thread started")
    tracemalloc.start()
    while True:
        time.sleep(5)
        snapshot = tracemalloc.take_snapshot()
        print("\n---\n\n")
        display_top(snapshot)

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
import reverb

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
class RacingGameENV(py_environment.PyEnvironment):

    def __init__(self,mj):
        self.mj = mj
        self.stall = 0
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=-15, maximum=15, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(96,), dtype=np.float32, name='observation')
        self._state = np.zeros((96,),dtype=np.float32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((96,),dtype=np.float32)
        self._episode_ended = False
        self.mj.reload()
        return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        driver_args,vehicle_state = self.mj.step(*action)
        radar,state = driver_args
        self._state = np.concatenate((radar, [
            *state.velocity,
            state.yaw,
            state.pitch,
            state.roll])).astype(np.float32)
        if vehicle_state.progress == 0:
            self.stall += 1
        if self._episode_ended or self.mj.steps > 500*5 or self.stall > 60*3 or not vehicle_state.good_start:
            self.stall = 0
            reward = 0
            if not vehicle_state.good_start:
                reward = -100
            else:
                reward = ((vehicle_state.completion/100)+vehicle_state.laps)*(1 + (state.velocity[0]**2+state.velocity[1]**2)**0.5)
            return ts.termination(self._state, reward)
        
        stall_penalty = 1 - (vehicle_state.progress == 0 and self.stall > 60*2) * 0.5
        return ts.transition(self._state, reward=((state.velocity[0]**2+state.velocity[1]**2)**0.5)*stall_penalty*0.01, discount=.99)
print(tf.config.list_logical_devices('GPU'))
env_name = "FUckingCarman-v0" # @param {type:"string"}
num_iterations = 3000 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

fc_layer_params = (100,20)

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}


train = Mujoco("track")
train.run()
train_py_env = RacingGameENV(train)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)

eval_ = Mujoco("inkscape")
eval_.run()
eval_py_env = RacingGameENV(eval_)

eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
    train_step_counter=train_step_counter)
tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]
def collect_episode(environment, policy, num_episodes):

  driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      policy, use_tf_function=True),
    [rb_observer],
    max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      tf_agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
      replay_buffer_signature)
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)


tf_agent.train = common.function(tf_agent.train)


tf_agent.train_step_counter.assign(0)

avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  collect_episode(
      train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

  iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
  trajectories, _ = next(iterator)
  train_loss = tf_agent.train(experience=trajectories)

  replay_buffer.clear()

  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
