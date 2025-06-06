import logging
import socket
import sys
import threading #ドローンからのレスポンスを受信する
import time

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

DEFAULT_DISTANCE = 0.30  # メートル表記
DEFAULT_SPEED = 10
DEFAULT_DEGREE = 10


class DroneManager(object):
    def __init__(self, host_ip='192.168.10.2', host_port=8889,
                 drone_ip='192.168.10.1', drone_port=8889,
                 speed=DEFAULT_SPEED):
        self.host_ip = host_ip
        self.host_port = host_port
        self.drone_ip = drone_ip
        self.drone_port = drone_port
        self.drone_address = (drone_ip, drone_port)
        self.host_port = host_port
        self.speed = speed
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # socket for sending cmd
        self.socket.bind((self.host_ip, self.host_port))
        # self.socket.sendto(b'command', self.drone_address)
        # self.socket.sendto(b'steramon', self.drone_address)
        # self.send_command('command')
        # self.send_command('streamon')

        self.response = None
        self.stop_event = threading.Event()
        self._response_thread = threading.Thread(target=self.receive_response,
                                                 args=(self.stop_event,))
        self._response_thread.start()
        self.send_command('command')
        self.send_command('streamon')
        self.set_speed(self.speed)

    def receive_response(self, stop_event):
        while not stop_event.is_set():
            try:
                self.response, ip = self.socket.recvfrom(3000)
                logger.info({'action': 'receive_response',
                             'response': self.response})
            except socket.error as ex:
                logger.error({'action': 'receive_response',
                              'ex': ex})
                break

    def __dell__(self):
        self.stop()

    def stop(self):
        self.stop_event.set()
        retry = 0
        while self._response_thread.isAlive():
            time.sleep(0.6)
            if retry > 30:
                break
            retry += 1
        self.socket.close()

    def send_command(self, command):
        logger.info({'action': 'send_command', 'command': command})
        self.socket.sendto(command.encode('utf-8'), self.drone_address)

        retry = 0
        while self.response is None:
            time.sleep(0.5)
            if retry > 3:
                break
            retry += 1

        if self.response is None:
            response = None
        else:
            response = self.response.decode('utf-8')
        self.response = None
        return response

    def takeoff(self):
        return self.send_command('takeoff')

    def land(self):
        return self.send_command('land')

    def move(self, direction, distance):
        distance = float(distance)
        distance = int(round(distance * 100))
        return self.send_command(f'{direction} {distance}')

    def up(self, distance):
        return self.move('up', distance)

    def down(self, distance):
        return self.move('down', distance)

    def left(self, distance):
        return self.move('left', distance)

    def right(self, distance):
        return self.move('right', distance)

    def forward(self, distance):
        return self.move('forward', distance)

    def back(self, distance):
        return self.move('back', distance)

    def set_speed(self, speed):
        return self.send_command(f'speed {speed}')

    def clockwise(self, degree=DEFAULT_DEGREE):
        return self.send_command(f'cw {degree}')

    def counter_clockwise(self, degree=DEFAULT_DEGREE):
        return self.send_command(f'ccw {degree}')


if __name__ == '__main__':
    drone_manager = DroneManager()

    drone_manager.takeoff()
    time.sleep(10)
    drone_manager.up(0.3)
    time.sleep(5)
    drone_manager.set_speed(20)
    time.sleep(5)
    drone_manager.forward(0.5)
    time.sleep(5)
    drone_manager.clockwise(90)
    time.sleep(5)
    drone_manager.forward(0.5)
    time.sleep(5)
    drone_manager.clockwise(90)
    time.sleep(5)
    drone_manager.forward(0.5)
    time.sleep(5)
    drone_manager.clockwise(90)
    time.sleep(5)
    drone_manager.forward(0.5)
    time.sleep(5)

    drone_manager.land()
    drone_manager.stop()